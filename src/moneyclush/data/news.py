"""Headlines from public RSS feeds, grouped by the topic each dashboard tab cares about.

No API key involved — RSS is the original open feed format, still served
directly by every outlet below with no auth and no rate-limit tier to pay
for. That is also why X/Twitter is not here: its API stopped offering a
free tier in 2023, and a paid Bearer Token is a prerequisite this project
does not have yet.

Feeds mix RSS 2.0 and Atom, and pubDate formats differ (RFC 822 vs ISO
8601), so parsing normalises both into naive UTC datetimes for sorting.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime

import httpx
import structlog

log = structlog.get_logger()

UA_HEADERS = {"User-Agent": "Mozilla/5.0"}
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

FEEDS = {
    "crypto": [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Cointelegraph", "https://cointelegraph.com/rss"),
    ],
    "sports": [
        ("BBC Sport", "https://feeds.bbci.co.uk/sport/rss.xml"),
        ("ESPN", "https://www.espn.com/espn/rss/news"),
    ],
    "general": [
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("NPR", "https://feeds.npr.org/1001/rss.xml"),
        ("TechCrunch", "https://techcrunch.com/feed/"),
    ],
}


TRANSLATE_URL = "https://api.mymemory.translated.net/get"

# A headline stays in a feed for hours, so the same string would otherwise be
# re-translated every refresh. Keyed by source text, this keeps the daily call
# count in the low hundreds instead of the tens of thousands.
_TRANSLATION_CACHE: dict[str, str] = {}
_TRANSLATE_LIMIT = asyncio.Semaphore(4)


@dataclass
class Headline:
    title: str
    link: str
    source: str
    published: datetime | None
    title_original: str | None = None


async def _translate(client: httpx.AsyncClient, text: str) -> str:
    """English headline into Spanish, falling back to the original on failure.

    A missing translation is never worth failing the feed over, so every
    error path returns the source text unchanged.
    """
    if text in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[text]

    try:
        async with _TRANSLATE_LIMIT:
            resp = await client.get(
                TRANSLATE_URL,
                params={"q": text[:480], "langpair": "en|es"},
                timeout=8,
            )
        resp.raise_for_status()
        payload = resp.json()
        translated = (payload.get("responseData") or {}).get("translatedText") or ""
        # The API echoes the input back on quota exhaustion; treat that as a miss.
        if not translated or translated.strip().upper() == text.strip().upper():
            return text
        _TRANSLATION_CACHE[text] = translated
        return translated
    except Exception as exc:
        log.warning("news.translate_failed", error=str(exc)[:80])
        return text


async def translate_headlines(
    client: httpx.AsyncClient, headlines: list[Headline]
) -> list[Headline]:
    """Translate titles in place, keeping the original for reference."""
    translations = await asyncio.gather(
        *(_translate(client, h.title) for h in headlines)
    )
    for headline, translated in zip(headlines, translations):
        if translated != headline.title:
            headline.title_original = headline.title
            headline.title = translated
    return headlines


def _parse_date(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).replace(tzinfo=None)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _parse_feed(xml_bytes: bytes, source: str) -> list[Headline]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if title and link:
            items.append(Headline(title, link, source, _parse_date(item.findtext("pubDate") or "")))
    if items:
        return items

    for entry in root.findall(".//atom:entry", ATOM_NS):
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        link_el = entry.find("atom:link", ATOM_NS)
        link = link_el.get("href", "") if link_el is not None else ""
        pub = (
            entry.findtext("atom:published", default="", namespaces=ATOM_NS)
            or entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
            or ""
        )
        if title and link:
            items.append(Headline(title, link, source, _parse_date(pub)))
    return items


async def _fetch_one(client: httpx.AsyncClient, source: str, url: str) -> list[Headline]:
    try:
        resp = await client.get(
            url, headers=UA_HEADERS, timeout=10, follow_redirects=True
        )
        resp.raise_for_status()
        return _parse_feed(resp.content, source)
    except Exception as exc:
        log.warning("news.feed_failed", source=source, error=str(exc)[:100])
        return []


async def fetch_headlines(
    client: httpx.AsyncClient,
    category: str,
    limit: int = 12,
    translate: bool = True,
) -> list[Headline]:
    """Latest headlines for one category, newest first, deduped by title."""
    feeds = FEEDS.get(category, [])
    results = await asyncio.gather(*(_fetch_one(client, s, u) for s, u in feeds))

    seen: set[str] = set()
    merged: list[Headline] = []
    for headline in sorted(
        (h for batch in results for h in batch),
        key=lambda h: h.published or datetime.min,
        reverse=True,
    ):
        key = headline.title.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(headline)
        if len(merged) >= limit:
            break

    if translate and merged:
        merged = await translate_headlines(client, merged)
    return merged


def to_rows(headlines: list[Headline]) -> list[dict]:
    return [
        {
            "title": h.title,
            "title_original": h.title_original,
            "link": h.link,
            "source": h.source,
            "published": h.published.isoformat() if h.published else None,
        }
        for h in headlines
    ]
