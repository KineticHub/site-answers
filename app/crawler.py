"""Crawl one website: find its pages, read them, and store passages and form rules.

Pages are found from the sitemap and by following links. Each page is fetched plainly
first; if that yields little text, it is loaded in a real browser, and the difference is
recorded (a page that is empty without JavaScript is invisible to many crawlers).
robots.txt is honoured: disallowed pages are listed as blocked and never fetched.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import re
from collections import deque
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
from xml.etree.ElementTree import ParseError

import httpx
import tldextract
from defusedxml import DefusedXmlException
from defusedxml import ElementTree
from playwright.async_api import Browser, Error as PlaywrightError, async_playwright

from .extract import Extracted, describe_form, extract
from .store import FoundVia, PageRecord, PageStatus, Store

log = logging.getLogger("site-answers.crawler")

USER_AGENT = "SiteAnswersBot/0.1 (site audit run by the site owner)"
FETCH_TIMEOUT = 15.0            # seconds per plain request
RENDER_TIMEOUT_MS = 15_000      # per browser page load
CONCURRENCY = 4
MAX_SITEMAPS = 20
THIN_TEXT = 600                 # characters; below this a scripted page is re-read in a browser
MIN_UNTITLED_WORDS = 4
RENDER_GAIN = 200               # extra characters the browser must reveal to count as "needs JavaScript"
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".css", ".js", ".json", ".xml",
    ".zip", ".gz", ".mp4", ".mp3", ".mov", ".woff", ".woff2", ".ttf", ".doc", ".docx", ".xls", ".xlsx",
)
SITEMAP_LOC_TAGS = frozenset({"{http://www.sitemaps.org/schemas/sitemap/0.9}loc", "loc"})
TRACKING_PARAMS = re.compile(r"^(utm_|fbclid$|gclid$|mc_cid$|mc_eid$|ref$)")


@dataclass(frozen=True)
class Scope:
    """Which URLs belong to the site being crawled."""

    key: str          # what is stored: "host:port" for local/IP sites, registrable domain otherwise
    exact_host: bool  # True: only this exact host[:port]; False: the domain and all its subdomains

    def contains(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or parts.hostname is None:
            return False
        if self.exact_host:
            return parts.netloc.lower() == self.key
        host = parts.hostname.lower()
        return host == self.key or host.endswith("." + self.key)


def scope_for(start_url: str) -> Scope:
    parts = urlsplit(start_url)
    host = (parts.hostname or "").lower()
    if host == "localhost" or host.endswith(".localhost") or _parses_as_ip(host):
        return Scope(key=parts.netloc.lower(), exact_host=True)
    # Bundled public-suffix list only: no network call just to split a hostname.
    registered = tldextract.TLDExtract(suffix_list_urls=())(host).top_domain_under_public_suffix
    if not registered:
        return Scope(key=parts.netloc.lower(), exact_host=True)
    return Scope(key=registered, exact_host=False)


def _parses_as_ip(host: str) -> bool:
    # ipaddress offers no non-raising check, so this is the one place a ValueError is the signal.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def site_id_for(scope: Scope) -> str:
    return re.sub(r"[^a-z0-9]+", "-", scope.key).strip("-")


def normalise(url: str) -> str:
    """One spelling per page: no fragment, no tracking parameters, lower-case host."""
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not TRACKING_PARAMS.match(k)])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", query, ""))


def certificate_problem(exc: httpx.HTTPError) -> str | None:
    """What is wrong with the site's certificate, or None when the failure was something else."""
    text = str(exc)
    if "CERTIFICATE_VERIFY_FAILED" not in text:
        return None
    if "certificate has expired" in text:
        return "it has expired"
    if "self-signed" in text or "self signed" in text:
        return "it is self-signed"
    if "Hostname mismatch" in text:
        return "it was issued for a different website"
    reason = text.split("certificate verify failed:")[-1].split("(_ssl")[0].strip() or "unknown reason"
    return f"it could not be verified ({reason})"


def describe_failure(exc: httpx.HTTPError) -> str:
    """Why a page could not be fetched, in words a site owner can act on."""
    problem = certificate_problem(exc)
    if problem is not None:
        return (
            f"The site's security certificate is invalid: {problem}. The page was not read. "
            "Browsers warn visitors away and crawlers refuse the site until it is fixed."
        )
    if isinstance(exc, httpx.TimeoutException):
        return f"The page took longer than {FETCH_TIMEOUT:.0f} seconds to respond."
    if isinstance(exc, httpx.ConnectError):
        return "Could not connect to the site."
    return f"Request failed: {exc!r}"


class Crawl:
    def __init__(self, store: Store, start_url: str, *, max_pages: int, allow_invalid_certificate: bool):
        self.store = store
        # The owner's explicit choice. Verification is always tried first; only a host whose
        # certificate actually fails is then read without it, and the site is flagged.
        self.allow_invalid_certificate = allow_invalid_certificate
        self._unverified_hosts: set[str] = set()
        self.start_url = normalise(start_url)
        self.scope = scope_for(self.start_url)
        self.site_id = site_id_for(self.scope)
        self.max_pages = max_pages
        self._queue: deque[tuple[str, FoundVia]] = deque()
        self._seen: set[str] = set()
        self._robots: dict[str, RobotFileParser | None] = {}
        self._robots_text: dict[str, str | None] = {}
        self._passage_hashes: set[str] = set()
        self._page_hashes: set[str] = set()
        self._fetched = 0
        self._pages_read = 0
        self._over_limit = 0
        self._browser: Browser | None = None
        self._browser_lock = asyncio.Lock()

    # -- entry point -------------------------------------------------------
    async def run(self) -> None:
        self.store.start_site(self.site_id, self.start_url, self.scope.key, self.max_pages)
        async with (
            self._new_client(verify=True) as client,
            self._new_client(verify=False) as unverified_client,
            async_playwright() as playwright,
        ):
            self._client = client
            self._unverified_client = unverified_client
            self._playwright = playwright
            await self._seed()
            workers = [asyncio.create_task(self._worker()) for _ in range(CONCURRENCY)]
            results = await asyncio.gather(*workers, return_exceptions=True)
            if self._browser is not None:
                await self._browser.close()
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            log.error("crawl of %s failed: %r", self.start_url, failures[0])
            self.store.finish_site(self.site_id, "failed", repr(failures[0]))
            raise failures[0]
        if self._pages_read == 0:
            start = self.store.page_by_url(self.site_id, self.start_url)
            reason = start["note"] if start and start["note"] else "No pages could be read."
            self.store.finish_site(self.site_id, "failed", reason)
            return
        self.store.set_over_limit(self.site_id, self._over_limit)
        self.store.finish_site(self.site_id, "done", None)

    def _new_client(self, *, verify: bool) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=FETCH_TIMEOUT, follow_redirects=True, verify=verify
        )

    async def _get(self, url: str) -> httpx.Response:
        """Fetch a URL with certificate checks, falling back to an unchecked read only if the owner allowed it."""
        host = urlsplit(url).netloc.lower()
        if host in self._unverified_hosts:
            return await self._unverified_client.get(url)
        try:
            return await self._client.get(url)
        except httpx.ConnectError as exc:
            problem = certificate_problem(exc)
            if problem is None or not self.allow_invalid_certificate:
                raise
            if not self._unverified_hosts:
                log.warning("reading %s without certificate checks, as the owner allowed: %s", host, problem)
                self.store.set_certificate_problem(self.site_id, problem)
            self._unverified_hosts.add(host)
            return await self._unverified_client.get(url)

    async def _seed(self) -> None:
        self._enqueue(self.start_url, "start")
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(self.start_url))
        robots = await self._robots_for(origin)
        self.store.set_robots(self.site_id, self._robots_text[origin])
        sitemaps = list(robots.site_maps() or []) if robots is not None else []
        if not sitemaps:
            sitemaps = [urljoin(origin, "/sitemap.xml")]
        for url in await self._sitemap_urls(sitemaps):
            self._enqueue(url, "sitemap")

    # -- frontier ----------------------------------------------------------
    def _enqueue(self, url: str, via: FoundVia) -> None:
        url = normalise(url)
        if url in self._seen or not self.scope.contains(url):
            return
        self._seen.add(url)
        if urlsplit(url).path.lower().endswith(SKIP_EXTENSIONS):
            return
        self._queue.append((url, via))

    async def _worker(self) -> None:
        idle_rounds = 0
        while idle_rounds < 20:  # other workers may still be adding links
            if not self._queue:
                idle_rounds += 1
                await asyncio.sleep(0.1)
                continue
            idle_rounds = 0
            url, via = self._queue.popleft()
            if self._fetched >= self.max_pages:
                self._over_limit += 1
                continue
            await self._visit(url, via)

    # -- one page ----------------------------------------------------------
    async def _visit(self, url: str, via: FoundVia) -> None:
        if not await self._allowed(url):
            self._save_unfetched(url, "blocked", via, "Disallowed by robots.txt")
            return
        self._fetched += 1

        try:
            response = await self._get(url)
        except httpx.HTTPError as exc:
            self._save_unfetched(url, "error", via, describe_failure(exc))
            return
        final_url = normalise(str(response.url))
        if final_url != url:
            if not self.scope.contains(final_url):
                self._save_unfetched(url, "skipped", via, f"Redirects off the site to {final_url}")
                return
            if final_url in self._seen:
                self._save_unfetched(url, "duplicate", via, f"Redirects to {final_url}")
                return
            self._seen.add(final_url)
        if response.status_code >= 400:
            self._save_page(
                final_url, status="error", via=via, http_status=response.status_code, page=None,
                needs_js=False, note=f"HTTP {response.status_code}", robots_header="",
            )
            return
        if "html" not in response.headers.get("content-type", "").lower():
            self._save_unfetched(final_url, "skipped", via, "Not a web page")
            return

        plain = extract(response.text, final_url)
        page, needs_js = plain, False
        if plain.text_len < THIN_TEXT and "<script" in response.text.lower():
            rendered_html = await self._render(final_url)
            if rendered_html is not None:
                rendered = extract(rendered_html, final_url)
                if rendered.text_len >= plain.text_len + RENDER_GAIN:
                    page, needs_js = rendered, True

        text_hash = hashlib.sha1("\n".join(s.text for s in page.sections).encode()).hexdigest()
        if text_hash in self._page_hashes:
            self._save_page(
                final_url, status="duplicate", via=via, http_status=response.status_code, page=page,
                needs_js=needs_js, note="Same text as another page", robots_header="",
            )
        else:
            self._page_hashes.add(text_hash)
            self._pages_read += 1
            notes = []
            if urlsplit(final_url).netloc.lower() in self._unverified_hosts:
                notes.append("Read despite the invalid security certificate")
            if needs_js:
                notes.append("Empty without JavaScript, so it was read in a browser")
            note = ". ".join(notes)
            page_id = self._save_page(
                final_url, status="ok", via=via, http_status=response.status_code, page=page,
                needs_js=needs_js, note=note, robots_header=response.headers.get("x-robots-tag", ""),
            )
            self._index(page_id, page)
        for link in page.links:
            self._enqueue(link, "link")

    def _index(self, page_id: int, page: Extracted) -> None:
        for section in page.sections:
            if not section.heading and len(section.text.split()) < MIN_UNTITLED_WORDS:
                continue  # a logo caption or lone link, not something a question could be answered from
            digest = hashlib.sha1(section.text.encode()).hexdigest()
            if digest in self._passage_hashes:
                continue  # menus and footers repeat on every page; keep one copy
            self._passage_hashes.add(digest)
            self.store.add_passage(
                site_id=self.site_id, page_id=page_id, heading=section.heading, text=section.text, kind="text"
            )
        for spec in page.forms:
            self.store.add_form(self.site_id, page_id, spec["name"], spec)
            if spec["signature"] in self._passage_hashes:
                continue
            self._passage_hashes.add(spec["signature"])
            self.store.add_passage(
                site_id=self.site_id, page_id=page_id, heading=f"Form rules: {spec['title'] or spec['name']}",
                text=describe_form(spec), kind="form",
            )

    def _save_page(
        self, url: str, *, status: PageStatus, via: FoundVia, http_status: int | None,
        page: Extracted | None, needs_js: bool, note: str, robots_header: str,
    ) -> int:
        rules = [page.robots_meta if page is not None else "", robots_header.strip().lower()]
        return self.store.save_page(self.site_id, PageRecord(
            url=url, status=status, found_via=via, http_status=http_status,
            title=page.title if page is not None else "",
            description=page.description if page is not None else "",
            needs_js=needs_js, text_len=page.text_len if page is not None else 0, note=note,
            page_date=page.page_date if page is not None else None,
            robots_meta=", ".join(rule for rule in rules if rule),
        ))

    def _save_unfetched(self, url: str, status: PageStatus, via: FoundVia, note: str) -> None:
        self._save_page(
            url, status=status, via=via, http_status=None, page=None, needs_js=False, note=note, robots_header="",
        )

    # -- fetching helpers --------------------------------------------------
    async def _get_text(self, url: str) -> str | None:
        """Body of a small text file, or None if it is missing or unreachable."""
        try:
            response = await self._get(url)
        except httpx.HTTPError as exc:
            log.warning("could not fetch %s: %r", url, exc)
            return None
        return response.text if response.status_code == 200 else None

    async def _robots_for(self, origin: str) -> RobotFileParser | None:
        if origin not in self._robots:
            text = await self._get_text(urljoin(origin, "/robots.txt"))
            parser = None
            if text is not None:
                parser = RobotFileParser()
                parser.parse(text.splitlines())
            self._robots[origin] = parser
            self._robots_text[origin] = text
        return self._robots[origin]

    async def _allowed(self, url: str) -> bool:
        origin = "{0.scheme}://{0.netloc}".format(urlsplit(url))
        robots = await self._robots_for(origin)
        return robots is None or robots.can_fetch(USER_AGENT, url)

    async def _sitemap_urls(self, sitemaps: list[str]) -> list[str]:
        """Page URLs from the sitemaps, following sitemap index files."""
        pages: list[str] = []
        pending = deque(sitemaps)
        read = 0
        while pending and read < MAX_SITEMAPS:
            text = await self._get_text(pending.popleft())
            read += 1
            if text is None:
                continue
            try:
                root = ElementTree.fromstring(text.encode())
            except (ParseError, DefusedXmlException) as exc:
                log.warning("unreadable sitemap: %r", exc)
                continue
            locs = [el.text.strip() for el in root.iter() if el.tag in SITEMAP_LOC_TAGS and el.text]
            if root.tag.endswith("sitemapindex"):
                pending.extend(locs)
            else:
                pages.extend(locs)
        return pages

    async def _render(self, url: str) -> str | None:
        """The page's HTML after its scripts have run, or None if the browser could not load it."""
        async with self._browser_lock:
            if self._browser is None:
                self._browser = await self._launch_browser()
            context = await self._browser.new_context(
                user_agent=USER_AGENT,
                ignore_https_errors=urlsplit(url).netloc.lower() in self._unverified_hosts,
            )
            try:
                page = await context.new_page()
                await page.goto(url, wait_until="networkidle", timeout=RENDER_TIMEOUT_MS)
                return await page.content()
            except PlaywrightError as exc:
                log.warning("browser could not load %s: %s", url, str(exc).splitlines()[0])
                return None
            finally:
                await context.close()

    async def _launch_browser(self) -> Browser:
        """Playwright's own Chromium if it is installed, otherwise the machine's Google Chrome."""
        try:
            return await self._playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            log.info("bundled Chromium unavailable (%s); using installed Chrome", str(exc).splitlines()[0])
            return await self._playwright.chromium.launch(headless=True, channel="chrome")
