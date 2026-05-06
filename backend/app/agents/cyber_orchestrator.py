from __future__ import annotations

from typing import Any, Dict, List

from app.agents.domain_orchestrator_base import DomainOrchestratorBase


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


class CyberSecurityOrchestrator(DomainOrchestratorBase):
    agent_type = "cyber_orchestrator"
    domain_label = "Cybersecurity"

    def select_agents(self, *, question: str, context: Dict[str, Any]) -> List[str]:
        q = question or ""
        picks: List[str] = ["qa_cyber"]

        # Networking specialization
        if _contains_any(
            q,
            [
                "network",
                "tcp",
                "udp",
                "dns",
                "dhcp",
                "routing",
                "router",
                "switch",
                "firewall",
                "vpn",
                "tls",
                "ssl",
                "http",
                "https",
                "wireshark",
                "packet",
                "subnet",
                "cidr",
                "bgp",
                "nat",
            ],
        ):
            picks.append("qa_cyber_networking")

        # Pentesting specialization (authorized)
        if _contains_any(
            q,
            [
                "pentest",
                "pen test",
                "penetration",
                "red team",
                "recon",
                "nmap",
                "burp",
                "metasploit",
                "exploit",
                "sql injection",
                "xss",
                "ssrf",
                "rce",
                "privilege escalation",
                "cve",
            ],
        ):
            picks.append("qa_pentesting")

        # Avoid spawning too many agents at once
        uniq: List[str] = []
        for p in picks:
            if p not in uniq:
                uniq.append(p)
        return uniq[:3]

# Backward-compatible alias for older imports
class CyberOrchestrator(CyberSecurityOrchestrator):
    pass