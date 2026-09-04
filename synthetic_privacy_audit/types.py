from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AttackStatus(str, Enum):
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(slots=True)
class AttackResult:
    attack: str
    status: AttackStatus
    metrics: dict[str, float | int | str | None] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

