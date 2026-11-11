"""Web server: the owner's page, the crawl and question APIs, and the widget script."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from functools import cache
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .assistant import answer_question, settings
from .crawler import Crawl, normalise, scope_for, site_id_for
from .gaps import build_report
from .jev import jev_settings
from .nudge import decide
from .store import SignalKind, Store, default_db_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("site-answers")

DEFAULT_MAX_PAGES = 200
RECENT_QUESTIONS = 50
FRICTION_ROWS = 50


@cache
def _static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = Store(default_db_path())
    interrupted = app.state.store.fail_interrupted_crawls()
    if interrupted:
        log.warning("%d crawl(s) were cut off by a restart and are marked as failed", interrupted)
    app.state.crawls = {}  # site_id -> running asyncio.Task
    app.state.reports = {}  # site_id -> running asyncio.Task
    yield


app = FastAPI(title="Site Answers", lifespan=lifespan)

# The widget runs on the owner's site and calls this server from there. For this local
# build any origin may call it; a hosted version must limit each site to its own origin.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


# -- request shapes ------------------------------------------------------------

class CrawlRequest(BaseModel):
    url: str = Field(max_length=2000)
    max_pages: int = Field(ge=1, le=2000)
    # Read the site even if its certificate fails verification. Off unless the owner asks.
    allow_invalid_certificate: bool


class FieldState(BaseModel):
    """One form field as the widget saw it. It never includes what the visitor typed."""

    name: str = Field(max_length=200)
    label: str = Field(max_length=300)
    type: str = Field(max_length=40)
    required: bool
    empty: bool
    valid: bool
    problems: list[str] = Field(max_length=12)
    hint: str = Field(max_length=300)
    customMessage: str = Field(max_length=300)


class FormState(BaseModel):
    name: str = Field(max_length=200)
    submitLabel: str = Field(max_length=200)
    submitDisabled: bool
    fields: list[FieldState] = Field(max_length=80)


class LastSubmit(BaseModel):
    secondsAgo: int = Field(ge=0)
    blocked: bool


class PageState(BaseModel):
    forms: list[FormState] = Field(max_length=10)
    errors: list[str] = Field(max_length=10)
    lastSubmit: LastSubmit | None


class AskRequest(BaseModel):
    host: str = Field(max_length=300)
    page_url: str = Field(max_length=2000)
    question: str = Field(min_length=1, max_length=1000)
    page_state: PageState | None


class SignalTarget(BaseModel):
    """What the visitor clicked. For a text field this is its label, never its contents."""

    tag: str = Field(max_length=40)
    role: str = Field(max_length=40)
    text: str = Field(max_length=120)
    label: str = Field(max_length=200)
    disabled: bool
    isSubmit: bool
    interactive: bool
    formName: str = Field(max_length=200)
    cursor: str = Field(max_length=40)


class Signal(BaseModel):
    kind: SignalKind
    count: int = Field(ge=1, le=100)
    withinMs: int = Field(ge=0, le=600_000)
    target: SignalTarget


class NudgeRequest(BaseModel):
    host: str = Field(max_length=300)
    page_url: str = Field(max_length=2000)
    signal: Signal
    page_state: PageState | None


# -- pages and script ----------------------------------------------------------

@app.get("/")
async def owner_page() -> FileResponse:
    return FileResponse(_static_dir() / "owner.html")


@app.get("/widget.js")
async def widget_script() -> FileResponse:
    return FileResponse(
        _static_dir() / "widget.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"}
    )


# -- crawl ---------------------------------------------------------------------

@app.post("/api/crawl")
async def start_crawl(body: CrawlRequest, request: Request) -> dict:
    url = body.url.strip()
    if "://" not in url:
        url = "https://" + url
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise HTTPException(400, "Enter a web address, such as https://example.com")

    site_id = site_id_for(scope_for(normalise(url)))
    crawls: dict[str, asyncio.Task] = request.app.state.crawls
    if site_id in crawls and not crawls[site_id].done():
        raise HTTPException(409, "This site is already being read.")

    crawl = Crawl(
        request.app.state.store, url,
        max_pages=body.max_pages, allow_invalid_certificate=body.allow_invalid_certificate,
    )
    task = asyncio.create_task(crawl.run())
    task.add_done_callback(_report_crawl_failure)
    crawls[site_id] = task
    return {"site_id": site_id}


def _report_crawl_failure(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception() is not None:
        log.error("crawl failed", exc_info=task.exception())


@app.get("/api/sites")
async def list_sites(request: Request) -> dict:
    store: Store = request.app.state.store
    return {"sites": [{**site, "counts": store.counts(site["id"])} for site in store.list_sites()]}


@app.get("/api/sites/{site_id}")
async def site_detail(site_id: str, request: Request) -> dict:
    store: Store = request.app.state.store
    site = store.get_site(site_id)
    if site is None:
        raise HTTPException(404, "That site has not been read yet.")
    return {
        "site": site,
        "counts": store.counts(site_id),
        "pages": store.pages(site_id),
        "questions": store.questions(site_id, RECENT_QUESTIONS),
        "friction": store.friction_summary(site_id, FRICTION_ROWS),
        "gap_report": store.get_gap_report(site_id),
        "ai": asdict(settings()),
        "jev_enabled": jev_settings().enabled,
    }


# -- gap report ------------------------------------------------------------------

@app.post("/api/sites/{site_id}/gap-report")
async def start_gap_report(site_id: str, request: Request) -> dict:
    """Build the owner's report on what stops AI search from reading and citing the site."""
    store: Store = request.app.state.store
    site = store.get_site(site_id)
    if site is None:
        raise HTTPException(404, "That site has not been read yet.")
    if site["status"] == "crawling":
        raise HTTPException(409, "Wait for the site to finish being read first.")
    reports: dict[str, asyncio.Task] = request.app.state.reports
    if site_id in reports and not reports[site_id].done():
        raise HTTPException(409, "The report is already being built.")
    reports[site_id] = asyncio.create_task(_build_report(store, site_id))
    return {"status": "building"}


async def _build_report(store: Store, site_id: str) -> None:
    try:
        await build_report(store, site_id)
    except Exception as exc:
        # Error boundary for a background job: the owner sees the failure on the page and the log has the trace.
        store.fail_gap_report(site_id, f"The report could not be built: {exc!r}")
        log.exception("gap report failed for %s", site_id)


# -- questions -----------------------------------------------------------------

@app.post("/api/ask")
async def ask(body: AskRequest, request: Request) -> dict:
    store: Store = request.app.state.store
    site = store.site_for_host(body.host.lower())
    if site is None:
        raise HTTPException(404, "This site has not been read yet, so there is nothing to answer from.")
    reply = await answer_question(
        store,
        site=site,
        question=body.question.strip(),
        page_url=normalise(body.page_url) if body.page_url else "",
        page_state=body.page_state.model_dump() if body.page_state is not None else None,
    )
    return asdict(reply)


# -- proactive help ------------------------------------------------------------

@app.post("/api/nudge")
async def nudge(body: NudgeRequest, request: Request) -> dict:
    """A visitor looks stuck: decide whether to offer a help note, and record the event."""
    store: Store = request.app.state.store
    site = store.site_for_host(body.host.lower())
    if site is None:
        raise HTTPException(404, "This site has not been read yet.")
    result = await decide(
        store,
        site=site,
        page_url=normalise(body.page_url),
        signal=body.signal.model_dump(),
        page_state=body.page_state.model_dump() if body.page_state is not None else None,
    )
    return {"show": result.show, "cause": result.cause, "message": result.message, "details": result.details}
