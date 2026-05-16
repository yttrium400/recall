from __future__ import annotations

import json
from pathlib import Path

from .session import Session

DEFAULT_DIR = Path(".recall/sessions")


def _sessions_dir(base: Path | None = None) -> Path:
    d = (base or Path.cwd()) / ".recall" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(session: Session, base: Path | None = None) -> Path:
    d = _sessions_dir(base)
    ts = session.started_at.replace(":", "-").replace("Z", "")
    filename = f"{ts}_{session.id[:8]}.json"
    path = d / filename
    path.write_text(json.dumps(session.to_dict(), indent=2))
    return path


def load(path: Path) -> Session:
    return Session.from_dict(json.loads(path.read_text()))


def list_sessions(base: Path | None = None) -> list[Path]:
    d = _sessions_dir(base)
    return sorted(d.glob("*.json"))
