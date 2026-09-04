from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from synthetic_privacy_audit.context import AttackContext
from synthetic_privacy_audit.types import AttackResult, AttackStatus


class AttackPrerequisiteError(RuntimeError):
    """Expected absence of data/model resources; reported as a skipped attack."""


class BaseAttack(ABC):
    name: str

    @abstractmethod
    def run(self, context: AttackContext) -> AttackResult:
        """Execute the attack. Raise AttackPrerequisiteError when not applicable."""

    def execute(self, context: AttackContext) -> AttackResult:
        try:
            return self.run(context)
        except AttackPrerequisiteError as exc:
            return AttackResult(self.name, AttackStatus.SKIPPED, reason=str(exc))
        except (ValueError, KeyError) as exc:
            return AttackResult(self.name, AttackStatus.SKIPPED, reason=str(exc))
        except Exception as exc:
            return AttackResult(
                self.name,
                AttackStatus.FAILED,
                reason=f"{type(exc).__name__}: {exc}",
            )

    def completed(
        self,
        metrics: dict[str, float | int | str | None],
        *,
        details: dict | None = None,
    ) -> AttackResult:
        return AttackResult(
            self.name,
            AttackStatus.COMPLETED,
            metrics=metrics,
            details=details or {},
        )

    @staticmethod
    def require_columns(context: AttackContext, columns: Iterable[str]) -> None:
        context.frames_with_columns(tuple(columns))

    @staticmethod
    def require_releases(context: AttackContext, count: int = 2) -> None:
        if len(context.all_releases) < count:
            raise AttackPrerequisiteError(
                f"Requires at least {count} synthetic releases; found {len(context.all_releases)}."
            )

    @staticmethod
    def require_generator(context: AttackContext) -> None:
        if context.generator is None:
            raise AttackPrerequisiteError(
                "Requires a configured SyntheticGenerator adapter to retrain releases."
            )

