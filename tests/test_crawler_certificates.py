"""The opt-in read of a site whose security certificate is invalid."""

import httpx
import pytest

from app.crawler import Crawl
from app.store import Store

START = "https://shop.test/"
EXPIRED = "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired (_ssl.c:1010)"
PAGES = {
    "/": '<html><head><title>Shop</title></head><body><h1>Shop</h1><p>We sell kettles.</p><a href="/about">About</a></body></html>',
    "/about": "<html><head><title>About</title></head><body><h1>About</h1><p>Open on weekdays.</p></body></html>",
}


def serve_pages(request: httpx.Request) -> httpx.Response:
    if request.url.path in PAGES:
        return httpx.Response(200, html=PAGES[request.url.path])
    return httpx.Response(404, text="not found")


def refuse_certificate(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError(EXPIRED, request=request)


def use_site(monkeypatch, *, certificate_valid: bool):
    """Stand in for a website: the checked connection works only when the certificate is valid."""
    used = {"verified": 0, "unverified": 0}

    def counted(handler, key):
        def wrapped(request):
            used[key] += 1
            return handler(request)
        return wrapped

    def new_client(self, *, verify: bool) -> httpx.AsyncClient:
        if verify:
            handler = counted(serve_pages if certificate_valid else refuse_certificate, "verified")
        else:
            handler = counted(serve_pages, "unverified")
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    monkeypatch.setattr(Crawl, "_new_client", new_client)
    return used


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


async def test_invalid_certificate_stops_the_read_by_default(store, monkeypatch):
    used = use_site(monkeypatch, certificate_valid=False)
    crawl = Crawl(store, START, max_pages=10, allow_invalid_certificate=False)

    await crawl.run()

    site = store.get_site(crawl.site_id)
    assert site["status"] == "failed"
    assert site["error"].startswith("The site's security certificate is invalid: it has expired.")
    assert site["certificate_problem"] is None
    assert used["unverified"] == 0, "nothing may be read unchecked without the owner's say-so"


async def test_owner_can_choose_to_read_it_anyway_and_the_site_is_flagged(store, monkeypatch):
    use_site(monkeypatch, certificate_valid=False)
    crawl = Crawl(store, START, max_pages=10, allow_invalid_certificate=True)

    await crawl.run()

    site = store.get_site(crawl.site_id)
    assert (site["status"], site["certificate_problem"]) == ("done", "it has expired")
    read = {p["url"]: p["note"] for p in store.pages(crawl.site_id) if p["status"] == "ok"}
    assert read == {
        "https://shop.test/": "Read despite the invalid security certificate",
        "https://shop.test/about": "Read despite the invalid security certificate",
    }
    assert store.search(crawl.site_id, ["kettles"], 3)[0]["url"] == "https://shop.test/"


async def test_the_switch_changes_nothing_for_a_site_with_a_valid_certificate(store, monkeypatch):
    used = use_site(monkeypatch, certificate_valid=True)
    crawl = Crawl(store, START, max_pages=10, allow_invalid_certificate=True)

    await crawl.run()

    site = store.get_site(crawl.site_id)
    assert (site["status"], site["certificate_problem"]) == ("done", None)
    assert used["unverified"] == 0, "a valid certificate must keep every request on the checked connection"
    assert all(p["note"] == "" for p in store.pages(crawl.site_id) if p["status"] == "ok")


SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url><loc>https://blog.test/</loc><image:image><image:loc>https://blog.test/uploads/photo.png</image:loc></image:image></url>
  <url><loc>https://blog.test/a</loc></url>
  <url><loc>https://blog.test/b</loc></url>
  <url><loc>https://blog.test/c</loc></url>
  <url><loc>https://blog.test/guide.pdf</loc></url>
</urlset>"""
SEARCH_FORM = '<form name="search"><input type="text" name="q" required><button type="submit">Search</button></form>'


def serve_blog(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/sitemap.xml":
        return httpx.Response(200, text=SITEMAP)
    if path in ("/", "/a", "/b", "/c"):
        return httpx.Response(200, html=f"<html><body><h1>Post {path}</h1><p>Words about {path}.</p>{SEARCH_FORM}</body></html>")
    return httpx.Response(404, text="not found")


async def test_real_site_clutter_is_kept_out_of_the_report(store, monkeypatch):
    monkeypatch.setattr(
        Crawl, "_new_client",
        lambda self, *, verify: httpx.AsyncClient(transport=httpx.MockTransport(serve_blog), follow_redirects=True),
    )
    crawl = Crawl(store, "https://blog.test/", max_pages=2, allow_invalid_certificate=False)

    await crawl.run()

    site = store.get_site(crawl.site_id)
    pages = store.pages(crawl.site_id)
    assert [p["status"] for p in pages] == ["ok", "ok"], "images, PDFs and over-limit pages get no rows"
    assert site["over_limit"] == 2, "the two unread posts are counted, not listed"
    assert store.counts(crawl.site_id)["forms"] == 1, "the search form on every page is one form"
    form_passages = [h for h in store.search(crawl.site_id, ["search form rules"], 10) if h["kind"] == "form"]
    assert len(form_passages) == 1
