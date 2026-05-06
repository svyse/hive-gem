from __future__ import annotations

from typing import Any, Dict, List
import re

from app.agents.domain_orchestrator_base import DomainOrchestratorBase
from app.core.config import settings


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _normalize(text)
    for raw_kw in keywords:
        kw = _normalize(raw_kw)
        if not kw:
            continue
        if " " in kw or "-" in kw or "_" in kw:
            if kw in t:
                return True
            continue
        if re.search(rf"(?<![a-z0-9_]){re.escape(kw)}(?![a-z0-9_])", t):
            return True
    return False


class InteractiveOrchestrator(DomainOrchestratorBase):
    """Interactive Q&A orchestrator for STEM/biology.

    For hosted backends we can fan out across a couple of specialist agents.
    For the local backend we keep selection conservative to avoid very slow or
    memory-heavy multi-agent local inference runs.
    """

    agent_type = "interactive_orchestrator"
    domain_label = "Interactive (STEM/Biology)"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        q = question or ""
        picks: List[str] = []

        biology_keywords = [
            "biology", "cell", "gene", "genetic", "protein", "enzyme", "dna", "rna",
            "microbe", "virus", "bacteria", "evolution", "ecology",
        ]
        math_keywords = [
            "math", "algebra", "calculus", "probability", "statistics", "statistical",
            "equation", "proof", "theorem", "linear", "matrix", "optimization",
        ]
        engineering_keywords = [
            "engineering", "mechanical", "electrical", "civil", "design", "control",
            "signal", "circuit", "cad", "thermodynamics", "fluid", "structural",
        ]
        technology_keywords = [
            "technology", "software", "computer", "database", "network", "cloud",
            "ai", "ml", "machine learning", "llm", "api", "frontend", "backend",
            "security", "cuda", "gpu", "nvidia", "kernel", "parallel computing",
            "model", "models", "checkpoint", "weights", "tokenizer", "transformer",
            "transformers", "huggingface", "qwen", "coder", "programming", "code",
            "python", "inference", "fine tune", "fine-tune", "finetune", "lora",
            "quantization", "prompt", "embedding",
        ]
        science_keywords = [
            "physics", "chemistry", "science", "experiment", "hypothesis", "quantum",
            "relativity", "reaction", "molecule", "astronomy",
        ]

        if _contains_any(q, biology_keywords):
            picks.append("qa_biology")
        if _contains_any(q, math_keywords):
            picks.append("qa_math")
        if _contains_any(q, engineering_keywords):
            picks.append("qa_engineering")
        if _contains_any(q, technology_keywords):
            picks.append("qa_computing")
        if _contains_any(q, science_keywords):
            picks.append("qa_science")

        is_local = str(getattr(settings, "llm_backend", "")).strip().lower() == "local"

        # Hosted backends can tolerate a broader fallback pair. Local should stay narrow.
        if not picks:
            picks = ["qa_computing"] if is_local else ["qa_computing", "qa_science"]

        # When a question is clearly about software/models/GPUs, prefer Computing only.
        if "qa_computing" in picks and (
            _contains_any(q, technology_keywords)
            or _contains_any(q, ["qwen", "cuda", "gpu", "nvidia", "transformer", "huggingface", "model", "coder"])
        ):
            return ["qa_computing"]

        # Keep local fan-out extremely conservative.
        if is_local:
            return picks[:1]

        return picks[:3]
