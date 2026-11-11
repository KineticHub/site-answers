"""SQLite storage: sites, pages, searchable passages, forms and the question log."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

SiteStatus = Literal["crawling", "done", "failed"]
PageStatus = Literal["ok", "blocked", "error", "skipped", "duplicate"]
FoundVia = Literal["start", "sitemap", "link"]
PassageKind = Literal["text", "form"]
Category = Literal["fact", "how_to", "whole_site", "navigation", "form_help", "off_topic"]
AnswerMode = Literal["ai", "search-only", "form-check"]
SignalKind = Literal["rage_click", "dead_click", "failed_submit"]
Cause = Literal["form_blocked", "dead_element", "no_feedback", "not_frustrated", "unknown"]
DecidedBy = Literal["rules", "jev", "none"]
ReportStatus = Literal["building", "done", "failed"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sites (
    id          TEXT PRIMARY KEY,
    start_url   TEXT NOT NULL,
    scope       TEXT NOT NULL,          -- host[:port] or registrable domain the crawl stays inside
    status      TEXT NOT NULL,          -- crawling | done | failed
    started_at  REAL NOT NULL,
    finished_at REAL,
    max_pages   INTEGER NOT NULL,
    robots_txt  TEXT,
    error       TEXT,
    certificate_problem TEXT,           -- set when the owner chose to read a site whose certificate is invalid
    over_limit  INTEGER NOT NULL DEFAULT 0  -- pages found but not read because the page limit was reached
);
CREATE TABLE IF NOT EXISTS pages (
    id          INTEGER PRIMARY KEY,
    site_id     TEXT NOT NULL,
    url         TEXT NOT NULL,
    status      TEXT NOT NULL,          -- ok | blocked | error | skipped | duplicate
    http_status INTEGER,
    title       TEXT,
    description TEXT,
    needs_js    INTEGER NOT NULL,       -- 1 when the page had no real text until rendered in a browser
    text_len    INTEGER NOT NULL,
    found_via   TEXT NOT NULL,          -- start | sitemap | link
    note        TEXT,
    fetched_at  REAL NOT NULL,
    page_date   TEXT,                   -- YYYY-MM-DD the page says it was last changed or published
    robots_meta TEXT NOT NULL DEFAULT '',   -- index rules from the page and its headers, e.g. "noindex"
    UNIQUE (site_id, url)
);
CREATE TABLE IF NOT EXISTS forms (
    id          INTEGER PRIMARY KEY,
    site_id     TEXT NOT NULL,
    page_id     INTEGER NOT NULL,
    name        TEXT,
    spec        TEXT NOT NULL           -- JSON: fields, rules, submit label
);
CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
    heading, text,
    site_id UNINDEXED, page_id UNINDEXED, kind UNINDEXED,
    tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS questions (
    id          INTEGER PRIMARY KEY,
    site_id     TEXT NOT NULL,
    asked_at    REAL NOT NULL,
    page_url    TEXT,
    question    TEXT NOT NULL,
    category    TEXT,
    covered     INTEGER,                -- 1 the site answered it, 0 it did not
    missing     TEXT,                   -- what the site would need to say
    answer      TEXT,
    sources     TEXT,                   -- JSON list of page URLs
    mode        TEXT                    -- ai | search-only | form-check
);
CREATE TABLE IF NOT EXISTS gap_reports (
    site_id     TEXT PRIMARY KEY,
    status      TEXT NOT NULL,          -- building | done | failed
    started_at  REAL NOT NULL,
    finished_at REAL,
    error       TEXT,
    report      TEXT                    -- JSON
);
CREATE TABLE IF NOT EXISTS friction_events (
    id          INTEGER PRIMARY KEY,
    site_id     TEXT NOT NULL,
    at          REAL NOT NULL,
    page_url    TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- rage_click | dead_click | failed_submit
    target      TEXT NOT NULL,          -- what was clicked, e.g. "Request booking (button)"
    cause       TEXT NOT NULL,          -- form_blocked | dead_element | no_feedback | not_frustrated | unknown
    decided_by  TEXT NOT NULL,          -- rules | jev | none
    shown       INTEGER NOT NULL,       -- 1 when the visitor was offered a help note
    details     TEXT NOT NULL           -- JSON list: the form problems on screen at that moment
);
"""

# Words too common to help ranking.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does", "for", "from", "how",
    "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "our", "that", "the", "their", "them",
    "there", "they", "this", "to", "was", "we", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your",
})

MAX_SEARCH_TERMS = 24
COMMON_TERM_SHARE = 0.15          # a word in more than this share of a site's passages is too common to rank by
MIN_PASSAGES_TO_JUDGE_COMMON = 20


def _quoted(term: str) -> str:
    """A search term as an FTS5 string, so punctuation in it cannot be read as query syntax."""
    return '"' + term.replace('"', "") + '"'


@cache
def default_db_path() -> Path:
    """Where the database lives: a data folder beside the app package."""
    return Path(__file__).resolve().parent.parent / "data" / "site-answers.db"


@dataclass(frozen=True)
class PageRecord:
    """One crawled (or deliberately not crawled) URL."""

    url: str
    status: PageStatus
    found_via: FoundVia
    http_status: int | None
    title: str
    description: str
    needs_js: bool
    text_len: int
    note: str
    page_date: str | None
    robots_meta: str


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            # Databases made before this column existed get it added.
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(sites)")}
            if "certificate_problem" not in columns:
                self._db.execute("ALTER TABLE sites ADD COLUMN certificate_problem TEXT")
            if "over_limit" not in columns:
                self._db.execute("ALTER TABLE sites ADD COLUMN over_limit INTEGER NOT NULL DEFAULT 0")
            page_columns = {row["name"] for row in self._db.execute("PRAGMA table_info(pages)")}
            if "page_date" not in page_columns:
                self._db.execute("ALTER TABLE pages ADD COLUMN page_date TEXT")
            if "robots_meta" not in page_columns:
                self._db.execute("ALTER TABLE pages ADD COLUMN robots_meta TEXT NOT NULL DEFAULT ''")
            self._db.commit()

    # -- small helpers -----------------------------------------------------
    def _run(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _one(self, sql: str, args: tuple = ()) -> dict | None:
        rows = self._all(sql, args)
        return rows[0] if rows else None

    def _count(self, sql: str, args: tuple) -> int:
        row = self._one(sql, args)
        assert row is not None, "COUNT(*) always returns a row"
        return row["n"]

    # -- sites -------------------------------------------------------------
    def start_site(self, site_id: str, start_url: str, scope: str, max_pages: int) -> None:
        """Begin (or restart) a crawl. A re-crawl replaces the site's earlier pages."""
        with self._lock:
            for table in ("pages", "forms", "passages"):
                self._db.execute(f"DELETE FROM {table} WHERE site_id = ?", (site_id,))
            self._db.execute(
                "INSERT OR REPLACE INTO sites (id, start_url, scope, status, started_at, max_pages) "
                "VALUES (?, ?, ?, 'crawling', ?, ?)",
                (site_id, start_url, scope, time.time(), max_pages),
            )
            self._db.commit()

    def finish_site(self, site_id: str, status: SiteStatus, error: str | None) -> None:
        self._run(
            "UPDATE sites SET status = ?, finished_at = ?, error = ? WHERE id = ?",
            (status, time.time(), error, site_id),
        )

    def fail_interrupted_crawls(self) -> int:
        """A crawl cannot survive a server restart: mark any left mid-read as failed, so it can be re-run."""
        cur = self._run(
            "UPDATE sites SET status = 'failed', finished_at = ?, error = ? WHERE status = 'crawling'",
            (time.time(), "The server restarted before the read finished. Read the site again."),
        )
        self._run(
            "UPDATE gap_reports SET status = 'failed', finished_at = ?, error = ? WHERE status = 'building'",
            (time.time(), "The server restarted before the report was finished. Build it again."),
        )
        return cur.rowcount

    def set_over_limit(self, site_id: str, count: int) -> None:
        self._run("UPDATE sites SET over_limit = ? WHERE id = ?", (count, site_id))

    def set_certificate_problem(self, site_id: str, problem: str) -> None:
        self._run("UPDATE sites SET certificate_problem = ? WHERE id = ?", (problem, site_id))

    def set_robots(self, site_id: str, robots_txt: str | None) -> None:
        self._run("UPDATE sites SET robots_txt = ? WHERE id = ?", (robots_txt, site_id))

    def get_site(self, site_id: str) -> dict | None:
        return self._one("SELECT * FROM sites WHERE id = ?", (site_id,))

    def site_for_host(self, host: str) -> dict | None:
        """The crawled site a page on `host` belongs to (exact host first, then parent domain)."""
        site = self._one("SELECT * FROM sites WHERE scope = ? ORDER BY started_at DESC LIMIT 1", (host,))
        if site:
            return site
        bare = host.split(":")[0]
        for row in self._all("SELECT * FROM sites ORDER BY started_at DESC"):
            if bare == row["scope"] or bare.endswith("." + row["scope"]):
                return row
        return None

    def list_sites(self) -> list[dict]:
        return self._all("SELECT * FROM sites ORDER BY started_at DESC")

    # -- pages -------------------------------------------------------------
    def save_page(self, site_id: str, page: PageRecord) -> int:
        cur = self._run(
            "INSERT OR REPLACE INTO pages (site_id, url, status, found_via, http_status, title, description, "
            "needs_js, text_len, note, fetched_at, page_date, robots_meta) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                site_id, page.url, page.status, page.found_via, page.http_status, page.title,
                page.description, int(page.needs_js), page.text_len, page.note, time.time(),
                page.page_date, page.robots_meta,
            ),
        )
        assert cur.lastrowid is not None, "INSERT always sets lastrowid"
        return cur.lastrowid

    def pages(self, site_id: str) -> list[dict]:
        return self._all("SELECT * FROM pages WHERE site_id = ? ORDER BY id", (site_id,))

    def page_by_url(self, site_id: str, url: str) -> dict | None:
        return self._one("SELECT * FROM pages WHERE site_id = ? AND url = ?", (site_id, url))

    def counts(self, site_id: str) -> dict[str, int]:
        rows = self._all("SELECT status, COUNT(*) AS n FROM pages WHERE site_id = ? GROUP BY status", (site_id,))
        out = {r["status"]: r["n"] for r in rows}
        out["needs_js"] = self._count(
            "SELECT COUNT(*) AS n FROM pages WHERE site_id = ? AND needs_js = 1", (site_id,)
        )
        # Rows saved before forms had a signature fall back to their full description.
        out["forms"] = self._count(
            "SELECT COUNT(DISTINCT COALESCE(json_extract(spec, '$.signature'), spec)) AS n FROM forms WHERE site_id = ?",
            (site_id,),
        )
        out["passages"] = self._count("SELECT COUNT(*) AS n FROM passages WHERE site_id = ?", (site_id,))
        return out

    # -- passages and forms --------------------------------------------------
    def add_passage(self, *, site_id: str, page_id: int, heading: str, text: str, kind: PassageKind) -> None:
        self._run(
            "INSERT INTO passages (heading, text, site_id, page_id, kind) VALUES (?, ?, ?, ?, ?)",
            (heading, text, site_id, page_id, kind),
        )

    def add_form(self, site_id: str, page_id: int, name: str, spec: dict) -> None:
        self._run(
            "INSERT INTO forms (site_id, page_id, name, spec) VALUES (?, ?, ?, ?)",
            (site_id, page_id, name, json.dumps(spec)),
        )

    def forms_on_page(self, site_id: str, page_id: int) -> list[dict]:
        rows = self._all("SELECT spec FROM forms WHERE site_id = ? AND page_id = ?", (site_id, page_id))
        return [json.loads(r["spec"]) for r in rows]

    def search(self, site_id: str, queries: list[str], limit: int) -> list[dict]:
        """Keyword search over the site's passages, best matches first."""
        terms: list[str] = []
        for q in queries:
            for word in re.findall(r"[\w$]+", q.lower()):
                if word not in STOPWORDS and len(word) > 1 and word not in terms:
                    terms.append(word)
        if not terms:
            return []
        terms = self._without_common_terms(site_id, terms[:MAX_SEARCH_TERMS])
        match = " OR ".join(_quoted(t) for t in terms)
        return self._all(
            """
            SELECT passages.rowid AS id, passages.heading, passages.text, passages.kind,
                   pages.url, pages.title, bm25(passages, 4.0, 1.0) AS score
            FROM passages JOIN pages ON pages.id = passages.page_id
            WHERE passages MATCH ? AND passages.site_id = ?
            ORDER BY score LIMIT ?
            """,
            (match, site_id, limit),
        )

    def _without_common_terms(self, site_id: str, terms: list[str]) -> list[str]:
        """Drop words found in a large share of the site's passages, such as its own name.

        They match almost everything, so they bury the passages that match the words that matter.
        """
        total = self._count("SELECT COUNT(*) AS n FROM passages WHERE site_id = ?", (site_id,))
        if len(terms) < 2 or total < MIN_PASSAGES_TO_JUDGE_COMMON:
            return terms
        rare = [
            t for t in terms
            if self._count(
                "SELECT COUNT(*) AS n FROM passages WHERE passages MATCH ? AND site_id = ?", (_quoted(t), site_id)
            ) <= total * COMMON_TERM_SHARE
        ]
        return rare or terms

    def passages_for_page(self, site_id: str, page_id: int) -> list[dict]:
        return self._all(
            """
            SELECT passages.rowid AS id, passages.heading, passages.text, passages.kind, pages.url, pages.title
            FROM passages JOIN pages ON pages.id = passages.page_id
            WHERE passages.site_id = ? AND passages.page_id = ? ORDER BY passages.rowid
            """,
            (site_id, page_id),
        )

    def all_passages(self, site_id: str) -> list[dict]:
        """Every text and form passage of a site, in reading order, with the page it came from."""
        return self._all(
            """
            SELECT passages.rowid AS id, passages.heading, passages.text, passages.kind,
                   CAST(passages.page_id AS INTEGER) AS page_id, pages.url, pages.title
            FROM passages JOIN pages ON pages.id = passages.page_id
            WHERE passages.site_id = ? ORDER BY passages.rowid
            """,
            (site_id,),
        )

    # -- gap report --------------------------------------------------------------
    def start_gap_report(self, site_id: str) -> None:
        self._run(
            "INSERT OR REPLACE INTO gap_reports (site_id, status, started_at) VALUES (?, 'building', ?)",
            (site_id, time.time()),
        )

    def finish_gap_report(self, site_id: str, report: dict) -> None:
        self._run(
            "UPDATE gap_reports SET status = 'done', finished_at = ?, report = ? WHERE site_id = ?",
            (time.time(), json.dumps(report), site_id),
        )

    def fail_gap_report(self, site_id: str, error: str) -> None:
        self._run(
            "UPDATE gap_reports SET status = 'failed', finished_at = ?, error = ? WHERE site_id = ?",
            (time.time(), error, site_id),
        )

    def get_gap_report(self, site_id: str) -> dict | None:
        row = self._one("SELECT * FROM gap_reports WHERE site_id = ?", (site_id,))
        if row is None:
            return None
        row["report"] = json.loads(row["report"]) if row["report"] else None
        return row

    def unanswered_questions(self, site_id: str, limit: int) -> list[dict]:
        """Questions real visitors asked that the site could not answer, most frequent first."""
        return self._all(
            """
            SELECT MIN(question) AS question, MAX(missing) AS missing, COUNT(*) AS times, MAX(page_url) AS page_url
            FROM questions WHERE site_id = ? AND covered = 0
            GROUP BY LOWER(TRIM(question)) ORDER BY times DESC, MAX(asked_at) DESC LIMIT ?
            """,
            (site_id, limit),
        )

    # -- question log (feeds the owner's gap report) -----------------------------
    def log_question(
        self,
        *,
        site_id: str,
        page_url: str,
        question: str,
        category: Category | None,
        covered: bool | None,
        missing: str,
        answer: str,
        sources: list[str],
        mode: AnswerMode,
    ) -> None:
        self._run(
            "INSERT INTO questions (site_id, asked_at, page_url, question, category, covered, missing, answer, "
            "sources, mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                site_id, time.time(), page_url, question, category,
                None if covered is None else int(covered), missing, answer, json.dumps(sources), mode,
            ),
        )

    # -- friction log (where visitors got stuck) ------------------------------
    def log_friction(
        self,
        *,
        site_id: str,
        page_url: str,
        kind: SignalKind,
        target: str,
        cause: Cause,
        decided_by: DecidedBy,
        shown: bool,
        details: list[str],
    ) -> None:
        self._run(
            "INSERT INTO friction_events (site_id, at, page_url, kind, target, cause, decided_by, shown, details) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (site_id, time.time(), page_url, kind, target, cause, decided_by, int(shown), json.dumps(details)),
        )

    def friction_summary(self, site_id: str, limit: int) -> list[dict]:
        """Stuck points grouped by page, action and cause, most frequent first, with the commonest reasons."""
        groups = self._all(
            """
            SELECT page_url, kind, target, cause, decided_by, COUNT(*) AS times, SUM(shown) AS helped,
                   MAX(at) AS last_at, json_group_array(json(details)) AS all_details
            FROM friction_events WHERE site_id = ?
            GROUP BY page_url, kind, target, cause, decided_by
            ORDER BY times DESC, last_at DESC LIMIT ?
            """,
            (site_id, limit),
        )
        for group in groups:
            tally: dict[str, int] = {}
            for details in json.loads(group.pop("all_details")):
                for reason in details:
                    tally[reason] = tally.get(reason, 0) + 1
            group["reasons"] = [
                {"text": text, "times": n} for text, n in sorted(tally.items(), key=lambda item: -item[1])[:3]
            ]
        return groups

    def questions(self, site_id: str, limit: int) -> list[dict]:
        return self._all(
            "SELECT * FROM questions WHERE site_id = ? ORDER BY asked_at DESC LIMIT ?", (site_id, limit)
        )
