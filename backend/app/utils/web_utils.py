from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, unquote, urljoin, urlparse, urlencode

from app.core.config import settings

try:
    import httpx  # type: ignore
except Exception as e:  # pragma: no cover
    httpx = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


class WebFetchError(RuntimeError):
    pass


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    # collapse excessive newlines/spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_from_html(html: str) -> str:
    """Best-effort HTML -> text extraction.

    Uses BeautifulSoup if available, otherwise falls back to a naive tag stripper.
    """
    if BeautifulSoup is None:
        # naive removal of tags
        text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return _normalize_whitespace(text)

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            pass

    text = soup.get_text(separator="\n")
    return _normalize_whitespace(text)


async def fetch_url(
    url: str,
    *,
    timeout_s: float | None = None,
    user_agent: str | None = None,
    max_bytes: int = 2_000_000,
) -> Dict[str, Any]:
    """Fetch a URL and return {url, status_code, content_type, text}.

    - Enforces a max_bytes cap to reduce risk of huge downloads.
    - Does *not* execute JS. For JS-heavy pages, you'll get partial content.
    """
    if httpx is None:
        raise WebFetchError("httpx is not installed; add it to requirements.txt (httpx>=0.27)")

    timeout_s = float(timeout_s or settings.web_request_timeout_s)
    user_agent = user_agent or settings.web_user_agent

    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}

    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
        try:
            resp = await client.get(url)
        except Exception as e:
            raise WebFetchError(f"Failed to fetch {url}: {e}") from e

    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    raw = resp.content[:max_bytes]
    encoding = resp.encoding or "utf-8"
    try:
        decoded = raw.decode(encoding, errors="ignore")
    except Exception:
        decoded = raw.decode("utf-8", errors="ignore")

    # Heuristic: treat as html if content-type suggests it or if it looks like it.
    if "html" in content_type or "<html" in decoded.lower() or "<body" in decoded.lower():
        text = extract_text_from_html(decoded)
    else:
        text = _normalize_whitespace(decoded)

    return {
        "url": str(resp.url),
        "status_code": resp.status_code,
        "content_type": content_type,
        "text": text,
    }



async def fetch_url_raw(
    url: str,
    *,
    timeout_s: float | None = None,
    user_agent: str | None = None,
    max_bytes: int = 2_000_000,
) -> Dict[str, Any]:
    """Fetch a URL and return {url, status_code, content_type, raw, text}.

    This is used by the WebScrapingAgent which may want to run CSS selectors
    against the original HTML (raw) while still getting an extracted text view.

    - Enforces a max_bytes cap to reduce risk of huge downloads.
    - Does *not* execute JS.
    """
    if httpx is None:
        raise WebFetchError("httpx is not installed; add it to requirements.txt (httpx>=0.27)")

    timeout_s = float(timeout_s or settings.web_request_timeout_s)
    user_agent = user_agent or settings.web_user_agent

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
        try:
            resp = await client.get(url)
        except Exception as e:
            raise WebFetchError(f"Failed to fetch {url}: {e}") from e

    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    raw_bytes = resp.content[:max_bytes]
    encoding = resp.encoding or "utf-8"
    try:
        decoded = raw_bytes.decode(encoding, errors="ignore")
    except Exception:
        decoded = raw_bytes.decode("utf-8", errors="ignore")

    # Heuristic: treat as html if content-type suggests it or if it looks like it.
    if "html" in content_type or "<html" in decoded.lower() or "<body" in decoded.lower():
        extracted_text = extract_text_from_html(decoded)
    else:
        extracted_text = _normalize_whitespace(decoded)

    return {
        "url": str(resp.url),
        "status_code": resp.status_code,
        "content_type": content_type,
        "raw": decoded,
        "text": extracted_text,
    }

def _decode_ddg_href(href: str) -> str:
    """DuckDuckGo result links are often redirect URLs with an 'uddg' param."""
    if not href:
        return href
    if href.startswith("/l/?") or "uddg=" in href:
        # make absolute if needed
        if href.startswith("/"):
            href = urljoin("https://duckduckgo.com", href)
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        if "uddg" in qs and qs["uddg"]:
            return unquote(qs["uddg"][0])
    return href


async def ddg_search(
    query: str,
    *,
    max_results: int = 5,
    timeout_s: float | None = None,
    user_agent: str | None = None,
) -> List[Dict[str, str]]:
    """Perform a lightweight DuckDuckGo HTML search.

    Note: This is *best effort* and can break if DDG changes HTML or blocks requests.
    """
    if httpx is None:
        raise WebFetchError("httpx is not installed; add it to requirements.txt (httpx>=0.27)")

    timeout_s = float(timeout_s or settings.web_request_timeout_s)
    user_agent = user_agent or settings.web_user_agent

    params = {"q": query}
    url = "https://duckduckgo.com/html/?" + urlencode(params)

    headers = {"User-Agent": user_agent, "Accept": "text/html"}

    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as e:
            raise WebFetchError(f"DuckDuckGo search failed: {e}") from e

    html = resp.text

    results: List[Dict[str, str]] = []

    if BeautifulSoup is None:
        # naive regex fallback for href + title
        for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.I | re.S):
            href = _decode_ddg_href(m.group(1))
            title = re.sub(r"<[^>]+>", "", m.group(2))
            title = _normalize_whitespace(title)
            if href and title:
                results.append({"title": title, "url": href, "snippet": ""})
            if len(results) >= max_results:
                break
        return results

    soup = BeautifulSoup(html, "html.parser")
    for a in soup.select("a.result__a"):
        href = a.get("href") or ""
        href = _decode_ddg_href(href)
        title = a.get_text(" ", strip=True)
        snippet = ""
        try:
            container = a.find_parent("div", class_=re.compile(r"\bresult\b"))
            if container:
                sn = container.select_one("a.result__snippet") or container.select_one("div.result__snippet") or container.select_one("span.result__snippet")
                if sn:
                    snippet = sn.get_text(" ", strip=True)
        except Exception:
            pass

        if href and title:
            results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= max_results:
            break

    return results
