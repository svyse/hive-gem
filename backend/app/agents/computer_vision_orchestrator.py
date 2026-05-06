from __future__ import annotations

from typing import Any, Dict, List

from app.agents.domain_orchestrator_base import DomainOrchestratorBase


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


class ComputerVisionOrchestrator(DomainOrchestratorBase):
    """Specialized orchestrator for computer vision.

    It routes to a CV specialist agent and can optionally include the
    general computing agent when the query is implementation-oriented
    (OpenCV, PyTorch, TensorFlow, etc.).
    """

    agent_type = "computer_vision_orchestrator"
    domain_label = "Computer vision"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        q = question or ""
        picks: List[str] = ["qa_computer_vision"]

        if _contains_any(
            q,
            [
                "opencv",
                "pytorch",
                "tensorflow",
                "keras",
                "onnx",
                "yolo",
                "detect",
                "segmentation",
                "object detection",
                "image processing",
                "camera",
                "video",
                "pipeline",
                "deployment",
                "api",
            ],
        ):
            picks.append("qa_computing")

        return picks[:3]
