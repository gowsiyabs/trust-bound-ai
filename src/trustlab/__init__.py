"""
AI Trust Lab - Dataset-agnostic stress-testing and safety demos for RAG systems.

Modules:
- hallucination_trigger : Trigger and detect hallucinations (empty/irrelevant context).
- context_poisoner      : Inject misleading or false context to test grounding.
- retrieval_stress      : Simulate retrieval failures + RetrievalTrustBoundary checks.
- red_team_probes       : Generic adversarial prompts (jailbreak, extraction, ambiguity).
- indirect_injection    : Indirect injection via poisoned document content.
- permission_audit      : False-negative / false-positive probes for the safety layer.
"""

from .hallucination_trigger import HallucinationTrigger
from .context_poisoner import ContextPoisoner
from .retrieval_stress import RetrievalStress, RetrievalTrustBoundary
from .red_team_probes import RedTeamProbes
from .indirect_injection import IndirectInjectionTester
from .permission_audit import PermissionAuditor

__all__ = [
    "HallucinationTrigger",
    "ContextPoisoner",
    "RetrievalStress",
    "RetrievalTrustBoundary",
    "RedTeamProbes",
    "IndirectInjectionTester",
    "PermissionAuditor",
]
