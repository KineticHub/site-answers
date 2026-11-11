"""Turn a page's HTML into searchable passages, outgoing links and form rules."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

DROP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe", "canvas", "head"}
HEADING_TAGS = {"h1", "h2", "h3", "h4"}
BLOCK_TAGS = {
    "p", "div", "section", "article", "main", "header", "footer", "nav", "aside", "ul", "ol", "li",
    "table", "tr", "form", "fieldset", "label", "blockquote", "pre", "dl", "dt", "dd", "br", "hr",
    "address", "figure", "figcaption", "details", "summary", "button", "select", "option", "textarea",
}
CHROME_TAGS = {"footer", "aside"}
MAX_PASSAGE = 900  # characters
DATE_PREFIX = re.compile(r"\d{4}-\d{2}-\d{2}")
# Where pages state when they were last changed, then when they were published.
MODIFIED_META = (("property", "article:modified_time"), ("property", "og:updated_time"), ("itemprop", "dateModified"), ("name", "last-modified"))
PUBLISHED_META = (("property", "article:published_time"), ("itemprop", "datePublished"), ("name", "date"))

log = logging.getLogger("site-answers.extract")


@dataclass(frozen=True)
class Section:
    heading: str
    text: str


@dataclass(frozen=True)
class Extracted:
    title: str
    description: str
    sections: list[Section]
    links: list[str]
    forms: list[dict]
    canonical: str | None
    text_len: int
    page_date: str | None   # YYYY-MM-DD the page says it was last changed (or published), if it says
    robots_meta: str        # the page's robots meta rules, such as "noindex", or ""


def extract(html: str, url: str) -> Extracted:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    meta = soup.find("meta", attrs={"name": "description"})
    description = meta["content"].strip() if meta and meta.get("content") else ""
    canonical_tag = soup.find("link", rel="canonical")
    canonical = urljoin(url, canonical_tag["href"]) if canonical_tag and canonical_tag.get("href") else None

    base = soup.find("base", href=True)
    base_url = urljoin(url, base["href"]) if base else url
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith(("#", "mailto:", "tel:", "javascript:", "data:")):
            links.append(urljoin(base_url, href))

    body = soup.body or soup
    sections = _sections(body)
    if not title:
        title = next((s.heading for s in sections if s.heading), "")
    return Extracted(
        title=title,
        description=description,
        sections=sections,
        links=links,
        forms=[_form_spec(f, soup) for f in body.find_all("form")],
        canonical=canonical,
        text_len=sum(len(s.text) for s in sections),
        page_date=_page_date(soup),
        robots_meta=_robots_meta(soup),
    )


def _robots_meta(soup: BeautifulSoup) -> str:
    rules = []
    for name in ("robots", "googlebot"):
        tag = soup.find("meta", attrs={"name": re.compile(f"^{name}$", re.IGNORECASE)})
        if tag and tag.get("content"):
            rules.append(tag["content"].strip().lower())
    return ", ".join(rules)


def _page_date(soup: BeautifulSoup) -> str | None:
    """When the page says it was last changed, or failing that published."""
    ld_modified, ld_published = _json_ld_dates(soup)
    candidates = [_meta_date(soup, MODIFIED_META), ld_modified, _meta_date(soup, PUBLISHED_META), ld_published]
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        candidates.append(time_tag["datetime"])
    for candidate in candidates:
        found = _as_date(candidate)
        if found is not None:
            return found
    return None


def _meta_date(soup: BeautifulSoup, names: tuple[tuple[str, str], ...]) -> str | None:
    for attr, value in names:
        tag = soup.find("meta", attrs={attr: value})
        if tag and tag.get("content"):
            return tag["content"]
    return None


def _json_ld_dates(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    """dateModified and datePublished from the page's structured data, if any."""
    modified = published = None
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        # json has no non-raising parser, and broken structured data is common on real sites.
        try:
            data = json.loads(script.string or "")
        except ValueError:
            log.debug("skipping structured data that is not valid JSON")
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                modified = modified or (node.get("dateModified") if isinstance(node.get("dateModified"), str) else None)
                published = published or (node.get("datePublished") if isinstance(node.get("datePublished"), str) else None)
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
    return modified, published


def _as_date(text: str | None) -> str | None:
    """The YYYY-MM-DD at the start of `text`, if it is a real calendar date."""
    match = DATE_PREFIX.match(text.strip()) if text else None
    if match is None:
        return None
    # date has no non-raising validator; "2023-13-45" matches the pattern but is not a date.
    try:
        date.fromisoformat(match.group())
    except ValueError:
        return None
    return match.group()


# -- text ------------------------------------------------------------------

def _sections(body: Tag) -> list[Section]:
    """Walk the page in reading order, starting a new section at each heading."""
    sections: list[Section] = []
    heading = ""
    lines: list[str] = []
    current: list[str] = []

    def end_line() -> None:
        text = " ".join(" ".join(current).split()).removesuffix(" |").strip()
        current.clear()
        if text:
            lines.append(text)

    def end_section() -> None:
        end_line()
        text = "\n".join(lines).strip()
        lines.clear()
        if text:
            for piece in _split(text):
                sections.append(Section(heading, piece))

    def walk(node: Tag) -> None:
        nonlocal heading
        for child in node.children:
            if isinstance(child, NavigableString):
                if type(child) is NavigableString:  # skip comments and doctype
                    current.append(str(child))
                continue
            if not isinstance(child, Tag) or child.name in DROP_TAGS or child.has_attr("hidden"):
                continue
            if child.name in HEADING_TAGS:
                end_section()
                heading = " ".join(child.get_text(" ").split())
                continue
            if child.name == "nav":
                continue  # menus are link lists, not content; their links are still followed
            if child.name in CHROME_TAGS:
                end_section()
                heading = ""
                walk(child)
                end_section()
                continue
            if child.name in ("td", "th"):
                walk(child)
                current.append(" | ")
                continue
            if child.name == "input":
                continue
            block = child.name in BLOCK_TAGS
            if block:
                end_line()
            walk(child)
            if block:
                end_line()

    walk(body)
    end_section()
    return sections


def _split(text: str) -> list[str]:
    """Keep passages short enough to quote: break long sections on line boundaries."""
    if len(text) <= MAX_PASSAGE:
        return [text]
    pieces, buf = [], ""
    for line in text.split("\n"):
        if buf and len(buf) + len(line) + 1 > MAX_PASSAGE:
            pieces.append(buf)
            buf = line
        else:
            buf = f"{buf}\n{line}" if buf else line
    if buf:
        pieces.append(buf)
    return pieces


# -- forms -----------------------------------------------------------------

def _form_spec(form: Tag, soup: BeautifulSoup) -> dict:
    """Record what a form asks for and the rules its fields enforce."""
    fields = []
    for el in form.find_all(["input", "select", "textarea"]):
        kind = (el.get("type") or el.name).lower() if el.name == "input" else el.name
        if kind in ("hidden", "submit", "button", "reset", "image"):
            continue
        rules = {}
        for attr in ("pattern", "minlength", "maxlength", "min", "max", "step", "accept"):
            if el.get(attr):
                rules[attr] = el[attr]
        # Limits that scripts enforce are often exposed as data- attributes.
        for attr, value in el.attrs.items():
            if attr.startswith("data-") and any(k in attr for k in ("max", "min", "size", "limit")):
                rules[attr] = value
        spec = {
            "name": el.get("name") or el.get("id") or "",
            "label": _label_for(el, form, soup),
            "type": kind,
            "required": el.has_attr("required"),
            "rules": rules,
            "hint": (el.get("title") or "").strip(),
        }
        hint_el = el.find_next_sibling(class_=lambda c: c and any(k in c for k in ("hint", "help")))
        if hint_el:
            visible = " ".join(hint_el.get_text(" ").split())
            spec["hint"] = f"{spec['hint']} {visible}".strip()
        if el.name == "select":
            spec["options"] = [" ".join(o.get_text(" ").split()) for o in el.find_all("option") if o.get("value") != ""]
        fields.append(spec)

    button = form.find("button", attrs={"type": "submit"}) or form.find("button") or form.find(
        "input", attrs={"type": "submit"}
    )
    submit_label = ""
    if button is not None:
        submit_label = " ".join(button.get_text(" ").split()) or button.get("value", "")
    heading = form.find_previous(list(HEADING_TAGS))
    # The same form (site search, newsletter sign-up) appears on many pages under different
    # headings. Its signature ignores where it sits, so it can be recognised as one form.
    signature = hashlib.sha1(json.dumps(
        [form.get("name") or form.get("id") or "", (form.get("method") or "get").lower(), submit_label, fields],
        sort_keys=True,
    ).encode()).hexdigest()
    return {
        "signature": signature,
        "name": form.get("name") or form.get("id") or (heading.get_text(" ").strip() if heading else "form"),
        "title": " ".join(heading.get_text(" ").split()) if heading else "",
        "method": (form.get("method") or "get").lower(),
        "submit_label": submit_label,
        "submit_starts_disabled": bool(button is not None and button.has_attr("disabled")),
        "browser_validation_off": form.has_attr("novalidate"),
        "fields": fields,
    }


def _label_for(el: Tag, form: Tag, soup: BeautifulSoup) -> str:
    if el.get("id"):
        label = soup.find("label", attrs={"for": el["id"]})
        if label:
            return " ".join(label.get_text(" ").split())
    parent = el.find_parent("label")
    if parent:
        return " ".join(parent.get_text(" ").split())
    return (el.get("aria-label") or el.get("placeholder") or el.get("name") or "").strip()


def describe_form(spec: dict) -> str:
    """Plain-language description of a form's rules, stored as a searchable passage."""
    lines = [f"Form: {spec.get('title') or spec['name']}. Submit button: \"{spec['submit_label']}\"."]
    if spec.get("submit_starts_disabled"):
        lines.append("The submit button is disabled when the page loads and only works once the form allows it.")
    for f in spec["fields"]:
        parts = [f"{f['label'] or f['name']} ({f['type']})", "required" if f["required"] else "optional"]
        rules = f["rules"]
        if "pattern" in rules:
            parts.append(f"must match the pattern {rules['pattern']}")
        if "minlength" in rules:
            parts.append(f"at least {rules['minlength']} characters")
        if "maxlength" in rules:
            parts.append(f"at most {rules['maxlength']} characters")
        if "min" in rules:
            parts.append(f"minimum {rules['min']}")
        if "max" in rules:
            parts.append(f"maximum {rules['max']}")
        if "accept" in rules:
            parts.append(f"accepted file types {rules['accept']}")
        for key, value in rules.items():
            if key.startswith("data-"):
                parts.append(f"{key[5:]} = {value}")
        if f.get("options"):
            parts.append("choices: " + ", ".join(f["options"]))
        if f["hint"]:
            parts.append(f"hint: {f['hint']}")
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines)
