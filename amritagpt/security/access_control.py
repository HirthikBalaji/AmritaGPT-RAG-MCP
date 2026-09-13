"""
Access Control and Prompt Injection Defense for AmritaGPT.
Enforces institutional role-based authorization (public, student, faculty, admin)
and neutralizes malicious prompt injection attempts embedded in document data.
"""
from typing import Set, Dict, Any, List
import re

from amritagpt.config import ROLE_HIERARCHY


class AccessControlManager:
    """Institutional Role-Based Access Control (RBAC)."""

    @classmethod
    def get_role_level(cls, role: str) -> int:
        return ROLE_HIERARCHY.get(role.lower().strip(), 1)

    @classmethod
    def is_authorized(cls, user_role: str, document_access_policy: str) -> bool:
        """Check if user role is sufficient to access the document."""
        user_level = cls.get_role_level(user_role)
        doc_level = cls.get_role_level(document_access_policy)
        return user_level >= doc_level

    @classmethod
    def filter_chunk_ids_by_role(cls, meta_store, user_role: str) -> Set[str]:
        """Pre-retrieval security filter: return set of chunk IDs user is authorized to see."""
        user_level = cls.get_role_level(user_role)
        
        # Allowed policy strings
        allowed_policies = [
            policy for policy, level in ROLE_HIERARCHY.items()
            if level <= user_level
        ]

        placeholders = ",".join(["?"] * len(allowed_policies))
        with meta_store._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT chunk_id FROM chunks WHERE access_policy IN ({placeholders})", allowed_policies)
            return {r[0] for r in cursor.fetchall()}


class PromptInjectionGuard:
    """Treats retrieved document content strictly as untrusted data, never as system instructions."""

    SUSPICIOUS_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"system\s+prompt",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"reveal\s+(api\s+key|password|confidential)",
        r"disregard\s+(the\s+above|rules)",
        r"print\s+system\s+instructions"
    ]

    @classmethod
    def sanitize_passage(cls, text: str) -> str:
        """Sanitize passages containing adversarial injection patterns."""
        sanitized = text
        for pat in cls.SUSPICIOUS_PATTERNS:
            sanitized = re.sub(pat, "[FILTERED_UNTRUSTED_INSTRUCTION]", sanitized, flags=re.IGNORECASE)
        return sanitized
