from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from app.agents.base import BaseAgent
from app.utils.web_utils import WebFetchError, fetch_url_raw

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore


WEB_SCRAPE_SYSTEM = """You are a Web Scraping agent inside a multi-agent hive.

You are given:
- a URL
- the page text (already fetched, no JS)
- an instruction about what to extract

Return ONLY valid JSON.

Constraints:
- Do NOT invent facts that are not in the provided page text.
- Prefer concise structured output.
- Include the URL in sources_used.
"""


def _is_disallowed_host(host: str) -> bool:
    """Best-effort SSRF guard for obvious local/private targets."""
    h = (host or "").strip().lower()
    if not h:
        return True
    if h in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    # IPv4 private ranges (string prefix checks; DNS resolution is out-of-scope here)
    if re.match(r"^127\.", h):
        return True
    if re.match(r"^10\.", h):
        return True
    if re.match(r"^192\.168\.", h):
        return True
    m = re.match(r"^172\.(\d+)\.", h)
    if m:
        try:
            b = int(m.group(1))
            if 16 <= b <= 31:
                return True
        except Exception:
            pass
    return False


def _parse_selector(sel: str) -> tuple[str, Optional[str]]:
    """Support 'css::attr(href)' syntax."""
    s = (sel or "").strip()
    m = re.search(r"::attr\(([^)]+)\)\s*$", s)
    if not m:
        return s, None
    attr = m.group(1).strip()
    css = s[: m.start()].strip()
    return css, attr


def _extract_node_value(node: Any, attr: Optional[str]) -> Optional[str]:
    if node is None:
        return None
    try:
        if attr:
            v = node.get(attr)
            if v is None:
                return None
            return str(v).strip()
        return str(node.get_text(" ", strip=True)).strip()
    except Exception:
        return None


class WebScrapingAgent(BaseAgent):
    """Specialized web scraping agent.

    Capabilities:
    - Fetch a page (HTML/text)
    - Extract structured fields via CSS selectors (when provided)
    - Or, do LLM-assisted extraction from the page text (instruction-only mode)
    """

    agent_type = "web_scraper"

    async def scrape(
        self,
        *,
        url: str,
        instruction: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
        list_selector: Optional[str] = None,
        item_fields: Optional[Dict[str, str]] = None,
        max_items: int = 20,
        include_text_preview: bool = True,
    ) -> Dict[str, Any]:
        self.set_state("scraping")
        self.log("scraping page")

        # Safety: only allow http(s) and avoid obvious private hosts
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            out = {"error": "Only http/https URLs are allowed", "url": url}
            self.latest_result = out
            self.set_state("idle")
            return out
        if _is_disallowed_host(parsed.hostname or ""):
            out = {"error": "Blocked URL host (private/local)", "url": url}
            self.latest_result = out
            self.set_state("idle")
            return out

        try:
            fetched = await fetch_url_raw(url)
        except WebFetchError as e:
            out = {"error": str(e), "url": url}
            self.latest_result = out
            self.set_state("idle")
            return out
        except Exception as e:
            out = {"error": f"fetch failed: {e}", "url": url}
            self.latest_result = out
            self.set_state("idle")
            return out

        raw = str(fetched.get("raw") or "")
        page_text = str(fetched.get("text") or "")
        final_url = str(fetched.get("url") or url)

        extracted: Dict[str, Any] = {}
        mode_used = ""

        # Selector-based extraction if requested
        if (fields or list_selector) and BeautifulSoup is not None:
            mode_used = "selectors"
            soup = BeautifulSoup(raw, "html.parser")

            # scalar fields
            if fields:
                for k, sel in fields.items():
                    css, attr = _parse_selector(sel)
                    node = soup.select_one(css) if css else None
                    extracted[k] = _extract_node_value(node, attr)

            # list extraction
            if list_selector:
                items = []
                nodes = soup.select(list_selector) if list_selector else []
                for n in nodes[: max(1, int(max_items))]:
                    if item_fields:
                        row: Dict[str, Any] = {}
                        for k, sel in item_fields.items():
                            css, attr = _parse_selector(sel)
                            node = n.select_one(css) if css else None
                            row[k] = _extract_node_value(node, attr)
                        items.append(row)
                    else:
                        items.append(_extract_node_value(n, None))
                extracted["items"] = items

        # Instruction-only extraction: use LLM on extracted page text
        if not extracted and instruction:
            mode_used = "llm_extract" if self.llm_available() else "text_only"
            if self.llm_available():
                # Pull in relevant sequence/memory patterns as "learning"
                mem_bundle = self.get_memory_bundle(instruction, include_type=True, include_hive=True, include_agent=False, limit_per_scope=4)
                memory_context = self.format_memory_bundle(mem_bundle)

                # Limit text to keep prompts bounded
                text_snippet = page_text[:16000]

                messages = [
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "url": final_url,
                                "instruction": instruction,
                                "page_text": text_snippet,
                                "memory_context": memory_context,
                                "required_output_schema": {
                                    "summary": "string",
                                    "data": "any (JSON)",
                                    "sources_used": ["https://..."],
                                    "confidence": "low|medium|high",
                                },
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    }
                ]
                try:
                    out = await self.ctx.llm.chat_json_async(system=WEB_SCRAPE_SYSTEM, messages=messages, temperature=0.2)
                    if isinstance(out, dict):
                        extracted = {
                            "summary": out.get("summary", ""),
                            "data": out.get("data"),
                            "sources_used": out.get("sources_used") or [final_url],
                            "confidence": out.get("confidence", "medium"),
                        }
                    else:
                        extracted = {"summary": "LLM returned non-object", "data": None, "sources_used": [final_url], "confidence": "low"}
                except Exception as e:
                    extracted = {"summary": f"LLM extraction failed: {e}", "data": None, "sources_used": [final_url], "confidence": "low"}
            else:
                extracted = {
                    "summary": "LLM unavailable; returning text preview only",
                    "data": {"text_preview": page_text[:4000]},
                    "sources_used": [final_url],
                    "confidence": "low",
                }

        out = {
            "url": final_url,
            "status_code": fetched.get("status_code"),
            "content_type": fetched.get("content_type"),
            "mode": mode_used or "raw_fetch",
            "extracted": extracted,
        }
        if include_text_preview:
            out["text_preview"] = page_text[:4000]

        self.latest_result = out

        # Persist to memory (type + hive)
        try:
            payload = {
                "kind": "web_scrape",
                "url": final_url,
                "instruction": instruction,
                "mode": out.get("mode"),
                "extracted": extracted,
            }
            self.add_type_memory(json.dumps(payload, ensure_ascii=False), tags=["web_scrape"], success=True if extracted else None)
            self.add_hive_memory(json.dumps(payload, ensure_ascii=False), tags=["web_scrape"], success=True if extracted else None)
        except Exception:
            pass

        self.set_state("idle")
        return out
