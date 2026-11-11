"""The owner's gap report: the rule-based checks, and how the AI-judged parts are assembled."""

from datetime import date

import pytest

from app import gaps
from app.assistant import AIUnavailable, Settings
from app.gaps import access_findings, build_report, freshness_findings
from app.store import PageRecord, Store

START = "https://shop.test/"
TODAY = date(2026, 11, 11)


def page(url, *, status="ok", needs_js=0, robots_meta="", page_date=None, note=""):
    return {
        "url": url, "title": url.rsplit("/", 1)[-1] or "Home", "status": status, "needs_js": needs_js,
        "robots_meta": robots_meta, "page_date": page_date, "note": note,
    }


def site(*, robots_txt=None, certificate_problem=None):
    return {"id": "shop", "scope": "shop.test", "start_url": START, "robots_txt": robots_txt, "certificate_problem": certificate_problem}


def titles(findings):
    return [(f.severity, f.title) for f in findings]


def test_blocking_a_search_crawler_is_serious_and_a_training_crawler_is_a_choice():
    robots = "User-agent: *\nDisallow: /admin/\n\nUser-agent: GPTBot\nDisallow: /\n\nUser-agent: PerplexityBot\nDisallow: /\n"

    found = access_findings(site(robots_txt=robots), [page(START), page(START + "prices")])

    assert titles(found) == [
        ("high", "robots.txt shuts out PerplexityBot (Perplexity)"),
        ("low", "robots.txt opts out of AI training by GPTBot (OpenAI)"),
    ]


def test_a_blanket_disallow_is_reported_for_every_kind_of_crawler():
    found = access_findings(site(robots_txt="User-agent: *\nDisallow: /\n"), [page(START)])

    assert [f.severity for f in found] == ["high", "medium", "low"]
    assert "Googlebot" in found[0].title and "OAI-SearchBot" in found[0].title and "Claude-SearchBot" in found[0].title


def test_an_ordinary_robots_file_raises_nothing():
    assert access_findings(site(robots_txt="User-agent: *\nDisallow: /cart/\n"), [page(START)]) == []


def test_page_level_obstacles_are_each_reported_with_their_pages():
    pages = [
        page(START),
        page(START + "prices", needs_js=1),
        page(START + "hidden", robots_meta="noindex, follow"),
        page(START + "quotes", robots_meta="index, nosnippet"),
        page(START + "broken", status="error", note="HTTP 500"),
        page(START + "admin", status="blocked"),
    ]

    found = access_findings(site(certificate_problem="it has expired"), pages)

    assert titles(found) == [
        ("high", "The site's security certificate is invalid"),
        ("high", "1 page tells search engines not to index it"),
        ("medium", "1 page forbids search engines from quoting it"),
        ("medium", "1 page is empty until JavaScript runs"),
        ("medium", "1 page failed to load"),
        ("low", "1 page is off limits to every crawler"),
    ]
    assert found[4].pages == [{"url": START + "broken", "title": "broken", "note": "HTTP 500"}]


def test_long_page_lists_are_cut_and_counted():
    pages = [page(f"{START}p{n}", needs_js=1) for n in range(gaps.MAX_LISTED_PAGES + 5)]

    [finding] = access_findings(site(), pages)

    assert (len(finding.pages), finding.more) == (gaps.MAX_LISTED_PAGES, 5)


def test_stale_and_undated_pages():
    pages = [
        page(START, page_date="2026-09-01"),
        page(START + "faq", page_date="2023-03-10"),
        page(START + "old", page_date="2021-01-05"),
        page(START + "a"), page(START + "b"), page(START + "c"), page(START + "d"),
    ]

    found = freshness_findings(pages, TODAY)

    assert titles(found) == [
        ("medium", "2 pages were last updated more than 18 months ago"),
        ("low", "4 of 7 pages do not say when they were last updated"),
    ]
    assert [p["note"] for p in found[0].pages] == ["Last updated 2021-01-05", "Last updated 2023-03-10"]


def test_recent_dated_pages_raise_nothing():
    assert freshness_findings([page(START, page_date="2026-10-01")], TODAY) == []


# -- assembling the AI-judged parts ---------------------------------------------

@pytest.fixture
def store(tmp_path):
    db = Store(tmp_path / "test.db")
    db.start_site("shop", START, "shop.test", 10)
    for path, heading, text in (
        ("", "Opening hours", "Saturday 10am to 4pm. Prices from $15."),
        ("about", "About", "We are the fastest in town. Open Saturdays 9am to 1pm."),
    ):
        page_id = db.save_page("shop", PageRecord(
            url=START + path, status="ok", found_via="start", http_status=200, title=heading, description="",
            needs_js=False, text_len=len(text), note="", page_date=None, robots_meta="",
        ))
        db.add_passage(site_id="shop", page_id=page_id, heading=heading, text=text, kind="text")
    db.finish_site("shop", "done", None)
    db.log_question(
        site_id="shop", page_url=START, question="Do you deliver?", category="fact", covered=False,
        missing="Whether delivery is offered.", answer="The site doesn't say.", sources=[], mode="ai",
    )
    return db


def use_ai(monkeypatch, *, replies=None, error=None):
    monkeypatch.setattr(gaps, "settings", lambda: Settings(ai_enabled=True, model="test-model"))

    async def fake_ask_model(*, system, user, schema, max_tokens, effort):
        if error is not None:
            raise error
        return replies[tuple(schema["properties"])]

    monkeypatch.setattr(gaps, "ask_model", fake_ask_model)


async def test_report_keeps_only_what_can_be_traced_to_a_real_page(store, monkeypatch):
    home, about = (p["id"] for p in store.all_passages("shop"))
    use_ai(monkeypatch, replies={
        ("business", "questions"): {"business": "A kettle shop.", "questions": [
            {"question": "When are you open on Saturday?", "search_terms": "Saturday hours"},
            {"question": "Do you sell teapots?", "search_terms": "teapots"},
        ]},
        ("results",): {"results": [
            {"index": 0, "verdict": "answered", "missing": "", "add_to_url": "", "source_ids": [home]},
            {"index": 1, "verdict": "missing", "missing": "Whether teapots are sold.", "add_to_url": START + "made-up-page", "source_ids": []},
            {"index": 7, "verdict": "missing", "missing": "A question nobody asked.", "add_to_url": START, "source_ids": []},
        ]},
        ("contradictions", "vague_claims"): {
            "contradictions": [
                {"topic": "Saturday hours", "first": {"passage_id": home, "quote": "Saturday 10am to 4pm"}, "second": {"passage_id": about, "quote": "Open Saturdays 9am to 1pm"}},
                {"topic": "Invented", "first": {"passage_id": 9999, "quote": "x"}, "second": {"passage_id": about, "quote": "y"}},
            ],
            "vague_claims": [{"passage_id": about, "claim": "the fastest in town", "evidence_needed": "A typical turnaround time."}],
        },
    })

    await build_report(store, "shop")

    row = store.get_gap_report("shop")
    ai = row["report"]["ai"]
    assert row["status"] == "done"
    assert [(q["question"], q["verdict"]) for q in ai["questions"]] == [
        ("Do you sell teapots?", "missing"),
        ("When are you open on Saturday?", "answered"),
    ], "unanswered first, and the out-of-range result dropped"
    assert ai["questions"][0]["add_to"] is None, "a page the site does not have is never suggested"
    assert ai["questions"][1]["sources"] == [{"url": START, "title": "Opening hours"}]
    assert [c["topic"] for c in ai["contradictions"]] == ["Saturday hours"]
    assert [s["url"] for s in ai["contradictions"][0]["statements"]] == [START, START + "about"]
    assert ai["vague_claims"] == [{"claim": "the fastest in town", "evidence_needed": "A typical turnaround time.", "url": START + "about", "title": "About"}]
    assert row["report"]["visitor_questions"][0]["question"] == "Do you deliver?"


async def test_rule_based_results_survive_an_ai_failure(store, monkeypatch):
    use_ai(monkeypatch, error=AIUnavailable("rate limited by the API"))

    await build_report(store, "shop")

    report = store.get_gap_report("shop")["report"]
    assert report["ai"] is None and "rate limited" in report["ai_note"]
    assert report["pages_read"] == 2 and report["freshness"][0]["title"].startswith("2 of 2 pages do not say")


async def test_without_a_key_the_report_says_so(store, monkeypatch):
    monkeypatch.setattr(gaps, "settings", lambda: Settings(ai_enabled=False, model="test-model"))

    await build_report(store, "shop")

    report = store.get_gap_report("shop")["report"]
    assert report["ai"] is None and report["ai_note"] == "The AI checks did not run, because no API key is set."


async def test_large_site_gets_a_second_wider_search_before_a_gap_is_reported(store, monkeypatch):
    monkeypatch.setattr(gaps, "WHOLE_SITE_CHARS", 0)  # treat the fixture as too large to read whole
    monkeypatch.setattr(gaps, "settings", lambda: Settings(ai_enabled=True, model="test-model"))
    home = store.all_passages("shop")[0]["id"]
    looks = []

    async def fake_ask_model(*, system, user, schema, max_tokens, effort):
        kind = tuple(schema["properties"])
        if kind == ("business", "questions"):
            return {"business": "A kettle shop.", "questions": [
                {"question": "When are you open on Saturday?", "search_terms": "Saturday hours"},
                {"question": "Do you sell teapots?", "search_terms": "teapots"},
            ]}
        if kind == ("contradictions", "vague_claims"):
            return {"contradictions": [], "vague_claims": []}
        looks.append(user)
        if len(looks) == 1:  # first look: the search missed the passage for question 0
            return {"results": [
                {"index": 0, "verdict": "missing", "missing": "Saturday opening hours.", "add_to_url": START, "source_ids": []},
                {"index": 1, "verdict": "missing", "missing": "Whether teapots are sold.", "add_to_url": START, "source_ids": []},
            ]}
        return {"results": [  # second look: found it
            {"index": 0, "verdict": "answered", "missing": "", "add_to_url": START, "source_ids": [home]},
            {"index": 1, "verdict": "missing", "missing": "Whether teapots are sold.", "add_to_url": START, "source_ids": []},
        ]}

    monkeypatch.setattr(gaps, "ask_model", fake_ask_model)

    await build_report(store, "shop")

    ai = store.get_gap_report("shop")["report"]["ai"]
    assert len(looks) == 2, "questions not found answered are searched for a second time"
    assert [(q["question"], q["verdict"]) for q in ai["questions"]] == [
        ("Do you sell teapots?", "missing"),
        ("When are you open on Saturday?", "answered"),
    ]
    assert ai["coverage"]["judged_against"] == "closest passages"


def test_search_is_not_swamped_by_the_site_name(tmp_path):
    db = Store(tmp_path / "test.db")
    db.start_site("shop", START, "shop.test", 10)
    page_id = db.save_page("shop", PageRecord(
        url=START, status="ok", found_via="start", http_status=200, title="Acme Kettles", description="",
        needs_js=False, text_len=0, note="", page_date=None, robots_meta="",
    ))
    for n in range(24):
        db.add_passage(site_id="shop", page_id=page_id, heading="Acme Kettles", text=f"Acme Kettles news item {n}.", kind="text")
    db.add_passage(site_id="shop", page_id=page_id, heading="Returns", text="Every kettle is guaranteed for two years.", kind="text")

    hits = db.search("shop", ["Is there a guarantee at Acme Kettles?"], 3)

    assert hits[0]["heading"] == "Returns"
