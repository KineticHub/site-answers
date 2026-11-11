"""Client for Jev, TypeSafe AI's decision model: typed questions in, structured answers out.

Jev returns judgments, never prose. It is used here for one cheap check: when a visitor's
clicks look like frustration but the page gives no clear cause, is it really frustration?

API reference: https://docs.typesafe.ai/api/system-one.md
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import httpx
from dotenv import load_dotenv

log = logging.getLogger("site-answers.jev")

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
TIMEOUT = 5.0  # seconds; a help note that arrives late is worse than none


@dataclass(frozen=True)
class JevSettings:
    enabled: bool
    url: str
    model: str


@cache
def jev_settings() -> JevSettings:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    return JevSettings(
        enabled=bool(os.environ.get("TYPESAFE_API_KEY")),
        url=os.environ.get("TYPESAFE_API_URL", DEFAULT_URL),
        model=os.environ.get("SITE_ANSWERS_JEV_MODEL", DEFAULT_MODEL),
    )


@cache
def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT)


class JevUnavailable(Exception):
    """Jev gave no usable answer; the caller treats the signal as undecided."""


async def ask_jev(*, state: dict, questions: dict[str, dict]) -> dict[str, dict]:
    """Ask Jev the given questions about `state`. Returns its answers keyed by question name."""
    settings = jev_settings()
    try:
        response = await _http_client().post(
            settings.url,
            headers={"Authorization": f"Bearer {os.environ.get('TYPESAFE_API_KEY', '')}"},
            json={"model": settings.model, "state": state, "questions": questions},
        )
    except httpx.HTTPError as exc:
        raise JevUnavailable(f"could not reach Jev: {exc!r}") from exc
    if response.status_code != 200:
        raise JevUnavailable(f"Jev returned HTTP {response.status_code}: {response.text[:200]}")

    answers = response.json().get("answers")
    if not isinstance(answers, dict) or not all(name in answers for name in questions):
        raise JevUnavailable("Jev's reply did not answer every question")
    return answers
