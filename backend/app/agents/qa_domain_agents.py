from __future__ import annotations

from app.agents.qa_base import DomainQAAAgent


# --- Interactive STEM Q&A agents ---


class ScienceQAAgent(DomainQAAAgent):
    agent_type = "qa_science"
    domain_name = "Science"


class ComputingQAAgent(DomainQAAAgent):
    agent_type = "qa_computing"
    domain_name = "Computing & AI"


class EngineeringQAAgent(DomainQAAAgent):
    agent_type = "qa_engineering"
    domain_name = "Engineering"


class MathQAAgent(DomainQAAAgent):
    agent_type = "qa_math"
    domain_name = "Mathematics"


class BiologyQAAgent(DomainQAAAgent):
    agent_type = "qa_biology"
    domain_name = "Biology"


# --- Specialist domains requested ---


class HRAgent(DomainQAAAgent):
    agent_type = "qa_hr"
    domain_name = "Human Resources"


class LawIndiaAgent(DomainQAAAgent):
    agent_type = "qa_law_india"
    domain_name = "Law (India)"
    safety_disclaimer = (
        "This is general legal information, not legal advice. "
        "For important matters, consult a qualified lawyer in your jurisdiction."
    )


class LawInternationalAgent(DomainQAAAgent):
    agent_type = "qa_law_international"
    domain_name = "Law (International)"
    safety_disclaimer = "This is general legal information, not legal advice. Laws vary widely by country and context."


class FinanceAgent(DomainQAAAgent):
    agent_type = "qa_finance"
    domain_name = "Finance"
    safety_disclaimer = "This is general financial information, not investment advice. Consider speaking with a qualified financial advisor."


class EconomicsAgent(DomainQAAAgent):
    agent_type = "qa_economics"
    domain_name = "Economics"


class SocialDynamicsAgent(DomainQAAAgent):
    agent_type = "qa_social"
    domain_name = "Social dynamics & speech"


class MedicineAgent(DomainQAAAgent):
    agent_type = "qa_medicine"
    domain_name = "Medicine"
    safety_disclaimer = "This is general medical information, not a diagnosis. If you have urgent or serious symptoms, seek professional medical care."


class CyberSecurityAgent(DomainQAAAgent):
    agent_type = "qa_cyber"
    domain_name = "Cybersecurity"
    safety_disclaimer = "This is defensive security guidance. Do not use it for wrongdoing; always follow applicable laws and policies."


class WebDesignAgent(DomainQAAAgent):
    agent_type = "qa_web_design"
    domain_name = "Web design (UI/UX, HTML/CSS, accessibility)"


class CyberNetworkingAgent(DomainQAAAgent):
    agent_type = "qa_cyber_networking"
    domain_name = "Cybersecurity (Networking)"
    safety_disclaimer = (
        "This is defensive networking/security guidance. Do not use it for wrongdoing; always follow applicable laws and policies."
    )


class PentestingAgent(DomainQAAAgent):
    agent_type = "qa_pentesting"
    domain_name = "Cybersecurity (Pentesting / authorized assessments)"
    safety_disclaimer = (
        "For authorized security testing only. Do not attempt to access systems you do not own or have explicit permission to test."
    )


class ComputerVisionAgent(DomainQAAAgent):
    agent_type = "qa_computer_vision"
    domain_name = "Computer vision"

