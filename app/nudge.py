"""Proactive help: decide whether a visitor who looks stuck should be offered a note.

The widget spots candidates with free rules (rapid clicks, clicks on a switched-off
control, repeated failed submits). This module decides what to do with one:

1. If a form on the page is refusing input, the cause is known. No model is needed.
2. Otherwise the signal is ambiguous (people also triple-click to select text), so Jev
   judges whether it is real frustration and which kind.
3. Without Jev, an ambiguous signal is recorded for the owner but nothing is shown.

Jev returns labels, not text, so every note comes from a template here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import get_args

from .assistant import form_findings
from .jev import JevUnavailable, ask_jev, jev_settings
from .store import Cause, DecidedBy, SignalKind, Store

log = logging.getLogger("site-answers.nudge")

MAX_DETAILS = 4            # problems listed in one note
FRUSTRATION_THRESHOLD = 0.6  # Jev's probability that the visitor is stuck
BUTTON_OFF_MARKER = "button is switched off"

JEV_QUESTIONS = {
    "frustrated": {
        "type": "noul",
        "instructions": "Is this website visitor likely stuck or frustrated, rather than using the page normally?",
        "criteria": {
            "true": "Repeated clicks that get no result, such as on a control that does nothing",
            "false": "Ordinary use, such as selecting text, stepping through a gallery, or a control meant to be clicked repeatedly",
        },
    },
    "cause": {
        "type": "choice",
        "instructions": "What best explains the visitor's repeated clicks?",
        "criteria": {
            "dead_element": "They clicked something that looks clickable but is not a link or button",
            "no_feedback": "The control is real but gives no visible response, so they keep clicking",
            "form_blocked": "A form is refusing to submit",
            "not_frustrated": "Normal use of the page, not a problem",
        },
    },
}


@dataclass(frozen=True)
class Nudge:
    show: bool
    cause: Cause
    decided_by: DecidedBy
    message: str
    details: list[str]


async def decide(
    store: Store, *, site: dict, page_url: str, signal: dict, page_state: dict | None
) -> Nudge:
    """Judge one frustration signal, record it for the owner, and say whether to show a note."""
    target = signal["target"]
    findings = form_findings(page_state)
    in_form = bool(target["formName"]) or signal["kind"] == "failed_submit"

    if in_form and findings:
        nudge = _form_note(signal["kind"], target, findings)
    elif jev_settings().enabled:
        nudge = await _ask_jev(signal, page_state)
    else:
        nudge = Nudge(show=False, cause="unknown", decided_by="none", message="", details=[])

    store.log_friction(
        site_id=site["id"], page_url=page_url, kind=signal["kind"], target=_target_name(target),
        cause=nudge.cause, decided_by=nudge.decided_by, shown=nudge.show, details=findings,
    )
    return nudge


def _form_note(kind: SignalKind, target: dict, findings: list[str]) -> Nudge:
    blockers = [f for f in findings if BUTTON_OFF_MARKER not in f]
    if kind == "dead_click" and target["disabled"]:
        name = f"\"{target['text']}\"" if target["text"] else "That button"
        message = f"{name} is switched off until the form is complete. Still needed:"
    else:
        message = "The form isn't going through yet. Here is what to fix:"
    details = blockers[:MAX_DETAILS]
    if len(blockers) > MAX_DETAILS:
        details.append(f"and {len(blockers) - MAX_DETAILS} more")
    if not blockers:
        message = "That button is switched off at the moment, so clicking it does nothing."
    return Nudge(show=True, cause="form_blocked", decided_by="rules", message=message, details=details)


async def _ask_jev(signal: dict, page_state: dict | None) -> Nudge:
    target = signal["target"]
    state = {
        "signal": {"kind": signal["kind"], "clicks": signal["count"], "within_ms": signal["withinMs"]},
        "element": {
            "tag": target["tag"], "role": target["role"], "text": target["text"],
            "is_link_or_button": target["interactive"], "disabled": target["disabled"],
            "mouse_cursor": target["cursor"], "inside_form": bool(target["formName"]),
        },
        "page": {
            "forms": len(page_state["forms"]) if page_state else 0,
            "error_messages_showing": page_state["errors"] if page_state else [],
        },
    }
    try:
        answers = await ask_jev(state=state, questions=JEV_QUESTIONS)
    except JevUnavailable as exc:
        log.error("Jev check failed, leaving the signal undecided: %s", exc)
        return Nudge(show=False, cause="unknown", decided_by="none", message="", details=[])

    probability = answers["frustrated"]["noul"]
    cause = answers["cause"]["choice"]
    if cause not in get_args(Cause):
        log.error("Jev returned a cause outside the offered choices: %r", cause)
        return Nudge(show=False, cause="unknown", decided_by="none", message="", details=[])
    if probability < FRUSTRATION_THRESHOLD or cause == "not_frustrated":
        return Nudge(show=False, cause="not_frustrated", decided_by="jev", message="", details=[])
    return Nudge(show=True, cause=cause, decided_by="jev", message=_jev_note(cause, target), details=[])


def _jev_note(cause: Cause, target: dict) -> str:
    name = f"\"{target['text']}\"" if target["text"] else "That"
    if cause == "dead_element":
        return f"{name} isn't a link or button, so clicking it does nothing. Tell me what you're after and I'll point you to it."
    if cause == "no_feedback":
        return f"{name} may be working without showing it. Tell me what you're trying to do and I'll check."
    return "It looks like something isn't working. Tell me what you're trying to do and I'll help."


def _target_name(target: dict) -> str:
    """How the owner's report names what was clicked."""
    label = target["text"] or target["label"]
    return f"{label} ({target['tag']})" if label else target["tag"]
