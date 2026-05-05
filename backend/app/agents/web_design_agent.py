from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from app.agents.base import BaseAgent


WEB_DESIGN_SYSTEM = """You are a Web Design agent inside a multi-agent hive.

Your job:
- Turn a product/feature goal into a practical web design + implementation outline.
- Prefer accessible, responsive, modern UI patterns.

Return ONLY valid JSON (no markdown fences).

Constraints:
- Keep suggestions implementable with common web stacks.
- If asked for code, include short snippets only.
- Reference prior hive/type memories when relevant.
"""


class WebDesignAgent(BaseAgent):
    """Specialized Web Design agent (UI/UX + frontend implementation guidance)."""

    agent_type = "web_design"

    async def design(
        self,
        *,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        stack: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.set_state("designing")
        self.log("generating web design")

        # Learning context
        mem_bundle = self.get_memory_bundle(goal, include_type=True, include_hive=True, include_agent=False, limit_per_scope=6)
        memory_context = self.format_memory_bundle(mem_bundle)

        if not self.llm_available():
            out = {
                "overview": "LLM unavailable (set OPENAI_API_KEY). Returning a placeholder web design outline.",
                "pages": [{"name": "Home", "purpose": "Entry page", "sections": ["Hero", "Features", "CTA"]}],
                "components": ["Navbar", "Footer", "CTAButton"],
                "style_guide": {"typography": "System UI", "spacing": "8px grid", "accessibility": ["Sufficient contrast", "Keyboard navigation"]},
                "implementation_steps": ["Set OPENAI_API_KEY", "Re-run to generate detailed design"],
                "snippets": [],
                "stack": stack or "unspecified",
            }
            self.latest_result = out
            try:
                payload = {"kind": "web_design", "goal": goal, "result": out}
                self.add_type_memory(json.dumps(payload, ensure_ascii=False), tags=["web_design"], success=None)
                self.add_hive_memory(json.dumps(payload, ensure_ascii=False), tags=["web_design"], success=None)
            except Exception:
                pass
            self.set_state("idle")
            return out

        messages = [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": goal,
                        "stack": stack,
                        "context": context or {},
                        "memory_context": memory_context,
                        "required_output_schema": {
                            "overview": "string",
                            "personas": [{"name": "string", "needs": ["string"], "pain_points": ["string"]}],
                            "information_architecture": {"sitemap": ["string"], "navigation": ["string"]},
                            "pages": [
                                {
                                    "name": "string",
                                    "purpose": "string",
                                    "sections": ["string"],
                                    "key_components": ["string"],
                                }
                            ],
                            "components": [
                                {
                                    "name": "string",
                                    "props": {"prop": "type"},
                                    "states": ["string"],
                                    "a11y_notes": ["string"],
                                }
                            ],
                            "style_guide": {
                                "typography": "string",
                                "spacing": "string",
                                "layout": "string",
                                "color_notes": "string",
                                "accessibility": ["string"],
                            },
                            "implementation_steps": ["string"],
                            "snippets": [
                                {"language": "html|css|js|tsx", "title": "string", "code": "string"}
                            ],
                            "confidence": "low|medium|high",
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            }
        ]

        out = await self.ctx.llm.chat_json_async(system=WEB_DESIGN_SYSTEM, messages=messages, temperature=0.2)
        if not isinstance(out, dict):
            raise ValueError("WebDesignAgent output must be a JSON object")

        out.setdefault("overview", "")
        out.setdefault("pages", [])
        out.setdefault("components", [])
        out.setdefault("style_guide", {})
        out.setdefault("implementation_steps", [])
        out.setdefault("snippets", [])
        out.setdefault("confidence", "medium")
        out.setdefault("stack", stack or "unspecified")

        self.latest_result = out

        # Persist learning
        try:
            payload = {"kind": "web_design", "goal": goal, "result": out}
            self.add_type_memory(json.dumps(payload, ensure_ascii=False), tags=["web_design"], success=True)
            self.add_hive_memory(json.dumps(payload, ensure_ascii=False), tags=["web_design"], success=True)
        except Exception:
            pass

        self.set_state("idle")
        return out
