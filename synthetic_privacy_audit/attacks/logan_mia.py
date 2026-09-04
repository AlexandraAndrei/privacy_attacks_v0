from __future__ import annotations

from synthetic_privacy_audit.attacks.base import AttackPrerequisiteError, BaseAttack
from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult


class LOGANMembershipInferenceAttack(BaseAttack):
    """Explicit integration point for LOGAN discriminator-based membership inference."""

    name = "logan_membership_inference"

    def run(self, context: AttackContext) -> AttackResult:
        del context
        raise AttackPrerequisiteError(
            "LOGAN requires a target or substitute GAN discriminator adapter. "
            "The generic SyntheticGenerator protocol intentionally exposes samples only."
        )

