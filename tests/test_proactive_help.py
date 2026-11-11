"""Checks for the stuck-visitor logic: what is diagnosed by rules, what goes to Jev, what is shown."""

import json

import httpx
import pytest

from app import jev, nudge
from app.assistant import form_findings
from app.crawler import describe_failure
from app.jev import DEFAULT_URL, JevSettings, JevUnavailable, ask_jev
from app.store import Store

SITE = {"id": "demo", "scope": "demo.test"}
PAGE = "https://demo.test/book"


def field(label, kind, problems, *, hint="", custom=""):
    return {
        "name": label.lower(), "label": label, "type": kind, "required": True, "empty": True,
        "valid": not problems, "problems": problems, "hint": hint, "customMessage": custom,
    }


def page_state(fields, *, submit_disabled):
    return {
        "forms": [{"name": "booking", "submitLabel": "Request booking", "submitDisabled": submit_disabled, "fields": fields}],
        "errors": [], "lastSubmit": None,
    }


def target(*, text, tag, disabled, form, interactive, cursor="pointer"):
    return {
        "tag": tag, "role": "", "text": text, "label": "", "disabled": disabled, "isSubmit": bool(form),
        "interactive": interactive, "formName": form, "cursor": cursor,
    }


def signal(kind, tgt):
    return {"kind": kind, "count": 3, "withinMs": 1000, "target": tgt}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


def use_jev(monkeypatch, *, enabled, answers=None, error=None):
    monkeypatch.setattr(nudge, "jev_settings", lambda: JevSettings(enabled=enabled, url=DEFAULT_URL, model="jev-latest"))

    async def fake_ask(*, state, questions):
        if error is not None:
            raise error
        return answers

    monkeypatch.setattr(nudge, "ask_jev", fake_ask)


def test_findings_put_the_least_obvious_problems_first():
    state = page_state(
        [
            field("Email", "email", ["valueMissing"]),
            field("Terms", "checkbox", ["valueMissing"]),
            field("Phone", "tel", ["patternMismatch"], hint="10 digits"),
            field("Date", "date", ["customError"], custom="Closed Sundays"),
        ],
        submit_disabled=True,
    )
    assert form_findings(state) == [
        "Date: Closed Sundays",
        "Phone is not in the expected format (expected: 10 digits).",
        'The box "Terms" is not ticked.',
        "Email is empty.",
        'The "Request booking" button is switched off, so clicking it does nothing.',
    ]


async def test_switched_off_button_is_diagnosed_by_rules_without_any_model(store, monkeypatch):
    use_jev(monkeypatch, enabled=True, error=AssertionError("Jev must not be called when the form explains it"))
    state = page_state([field("Terms", "checkbox", ["valueMissing"]), field("Email", "email", ["valueMissing"])], submit_disabled=True)
    tgt = target(text="Request booking", tag="button", disabled=True, form="booking", interactive=True)

    result = await nudge.decide(store, site=SITE, page_url=PAGE, signal=signal("dead_click", tgt), page_state=state)

    assert result.show and result.decided_by == "rules" and result.cause == "form_blocked"
    assert result.message.startswith('"Request booking" is switched off')
    assert result.details == ['The box "Terms" is not ticked.', "Email is empty."]
    [row] = store.friction_summary("demo", 10)
    assert (row["kind"], row["target"], row["helped"]) == ("dead_click", "Request booking (button)", 1)


async def test_long_problem_lists_are_cut_short(store, monkeypatch):
    use_jev(monkeypatch, enabled=False)
    state = page_state([field(f"Field {n}", "text", ["valueMissing"]) for n in range(7)], submit_disabled=False)
    tgt = target(text="Send", tag="button", disabled=False, form="booking", interactive=True)

    result = await nudge.decide(store, site=SITE, page_url=PAGE, signal=signal("failed_submit", tgt), page_state=state)

    assert len(result.details) == nudge.MAX_DETAILS + 1 and result.details[-1] == "and 3 more"


async def test_unclear_signal_without_jev_is_recorded_but_not_shown(store, monkeypatch):
    use_jev(monkeypatch, enabled=False)
    tgt = target(text="Call us now", tag="span", disabled=False, form="", interactive=False)

    result = await nudge.decide(store, site=SITE, page_url=PAGE, signal=signal("rage_click", tgt), page_state=None)

    assert not result.show and result.decided_by == "none" and result.cause == "unknown"
    [row] = store.friction_summary("demo", 10)
    assert (row["target"], row["helped"]) == ("Call us now (span)", 0)


async def test_jev_confirms_frustration_and_the_note_names_the_element(store, monkeypatch):
    use_jev(monkeypatch, enabled=True, answers={
        "frustrated": {"type": "noul", "noul": 0.93},
        "cause": {"type": "choice", "choice": "dead_element", "confidence": 0.8, "probabilities": {}},
    })
    tgt = target(text="Call us now", tag="span", disabled=False, form="", interactive=False)

    result = await nudge.decide(store, site=SITE, page_url=PAGE, signal=signal("rage_click", tgt), page_state=None)

    assert result.show and result.decided_by == "jev" and result.cause == "dead_element"
    assert result.message.startswith('"Call us now" isn\'t a link or button')


@pytest.mark.parametrize("answers", [
    {"frustrated": {"type": "noul", "noul": 0.2}, "cause": {"type": "choice", "choice": "dead_element"}},
    {"frustrated": {"type": "noul", "noul": 0.9}, "cause": {"type": "choice", "choice": "not_frustrated"}},
])
async def test_jev_can_say_it_is_ordinary_use(store, monkeypatch, answers):
    use_jev(monkeypatch, enabled=True, answers=answers)
    tgt = target(text="Next photo", tag="button", disabled=False, form="", interactive=True)

    result = await nudge.decide(store, site=SITE, page_url=PAGE, signal=signal("rage_click", tgt), page_state=None)

    assert not result.show and result.decided_by == "jev" and result.cause == "not_frustrated"


@pytest.mark.parametrize("failure", [
    {"error": JevUnavailable("timeout")},
    {"answers": {"frustrated": {"type": "noul", "noul": 0.9}, "cause": {"type": "choice", "choice": "made_up_label"}}},
])
async def test_a_failed_or_malformed_jev_reply_shows_nothing(store, monkeypatch, failure):
    use_jev(monkeypatch, enabled=True, **failure)
    tgt = target(text="Call us now", tag="span", disabled=False, form="", interactive=False)

    result = await nudge.decide(store, site=SITE, page_url=PAGE, signal=signal("rage_click", tgt), page_state=None)

    assert not result.show and result.decided_by == "none"


def use_jev_transport(monkeypatch, handler):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(jev, "jev_settings", lambda: JevSettings(enabled=True, url=DEFAULT_URL, model="jev-latest"))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(jev, "_http_client", lambda: client)


async def test_jev_request_matches_the_documented_shape(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "jev-1.14.0",
            "answers": {"frustrated": {"type": "noul", "noul": 0.98}},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        })

    use_jev_transport(monkeypatch, handler)
    questions = {"frustrated": {"type": "noul", "instructions": "Is the visitor stuck?"}}

    answers = await ask_jev(state={"clicks": 3}, questions=questions)

    assert seen == {
        "url": "https://api.typesafe.ai/v1/systemone",
        "auth": "Bearer test-key",
        "body": {"model": "jev-latest", "state": {"clicks": 3}, "questions": questions},
    }
    assert answers["frustrated"]["noul"] == 0.98


@pytest.mark.parametrize("response", [
    httpx.Response(429, json={"detail": "rate limit exceeded"}),
    httpx.Response(200, json={"model": "jev-1.14.0", "answers": {}}),
])
async def test_jev_errors_and_incomplete_replies_are_reported(monkeypatch, response):
    use_jev_transport(monkeypatch, lambda request: response)

    with pytest.raises(JevUnavailable):
        await ask_jev(state={}, questions={"frustrated": {"type": "noul", "instructions": "?"}})


def test_an_expired_certificate_is_explained_in_plain_words():
    exc = httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1010)")
    assert describe_failure(exc).startswith("The site's security certificate is invalid: it has expired.")
