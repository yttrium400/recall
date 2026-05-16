from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class Event:
    seq: int
    type: Literal["request", "response", "error"]
    timestamp: str
    payload: dict[str, Any]
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seq": self.seq,
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
        if self.duration_ms is not None:
            d["duration_ms"] = self.duration_ms
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        return cls(
            seq=d["seq"],
            type=d["type"],
            timestamp=d["timestamp"],
            payload=d["payload"],
            duration_ms=d.get("duration_ms"),
        )


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )
    provider: str = "anthropic"
    model: str = ""
    events: list[Event] = field(default_factory=list)

    def add_event(self, event: Event) -> None:
        self.events.append(event)

    def next_seq(self) -> int:
        return len(self.events) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "provider": self.provider,
            "model": self.model,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        s = cls(
            id=d["id"],
            started_at=d["started_at"],
            provider=d.get("provider", "anthropic"),
            model=d.get("model", ""),
        )
        s.events = [Event.from_dict(e) for e in d.get("events", [])]
        return s
