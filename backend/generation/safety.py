import re
from typing import Tuple, Optional

EMERGENCY_KEYWORDS = [
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bstroke\b",
    r"\bcan't breathe\b",
    r"\bcannot breathe\b",
    r"\bshortness of breath\b",
    r"\bdifficulty breathing\b",
    r"\bsevere bleeding\b",
    r"\bunconscious\b",
    r"\bseizure\b",
    r"\banaphylaxis\b",
    r"\bsuicid(e|al)\b",
    r"\bkill myself\b",
    r"\boverdose\b",
    r"\bpoisoning\b",
    r"\bcoughing up blood\b",
    r"\bsudden numbness\b",
    r"\bspeech slurred\b",
    r"\bfacial droop\b"
]

EMERGENCY_DISCLAIMER = """🚨 **EMERGENCY NOTICE**: If you or someone around you is experiencing a medical emergency (such as severe chest pain, sudden numbness/paralysis, difficulty breathing, severe bleeding, or thoughts of self-harm), please **immediately call 911** (or your local emergency services) or go to the nearest emergency department. Do not rely on an AI assistant for acute medical emergencies."""

GENERAL_HEALTHCARE_DISCLAIMER = """*Disclaimer: MediRAG provides evidence-based educational healthcare information extracted from verified clinical reference documents. It does not provide medical diagnoses, personalized treatment plans, or individualized drug prescriptions. Always consult a licensed healthcare professional for medical concerns.*"""

UNSUPPORTED_ANSWER_MESSAGE = """Based on the indexed medical reference documents, there is insufficient evidence to answer your question accurately. 

MediRAG strictly avoids fabricating medical information. Please provide relevant clinical documents or consult a qualified healthcare provider for guidance on this topic."""

class SafetyGuard:
    """
    Healthcare safety guardrails:
    - Detects acute medical emergencies
    - Enforces clinical disclaimers
    - Flags unsupported claims
    """

    @classmethod
    def check_emergency(cls, query: str) -> Tuple[bool, Optional[str]]:
        lower_q = query.lower()
        for pattern in EMERGENCY_KEYWORDS:
            if re.search(pattern, lower_q):
                return True, EMERGENCY_DISCLAIMER
        return False, None

    @classmethod
    def append_disclaimer(cls, answer: str, is_emergency: bool = False) -> str:
        parts = []
        if is_emergency:
            parts.append(EMERGENCY_DISCLAIMER)
            parts.append("---")
        parts.append(answer.strip())
        parts.append("---")
        parts.append(GENERAL_HEALTHCARE_DISCLAIMER)
        return "\n\n".join(parts)

    @classmethod
    def get_unsupported_response(cls) -> str:
        return f"{UNSUPPORTED_ANSWER_MESSAGE}\n\n---\n{GENERAL_HEALTHCARE_DISCLAIMER}"
