"""Audit-Log für Quellen-Entfernungen.

Wenn eine ECHTE Quelle aus der Wissensbasis entfernt wird (Index-Seiten-Cleanup,
manuelle Entfernung, …), MUSS das hier mit Begründung protokolliert werden —
nachvollziehbar und über /source-removals einsehbar. Leerer Müll/Orphans ohne
Inhalt brauchen das nicht.

Append-only JSONL: logs/source_removals.jsonl
"""
import json
import os
from datetime import datetime, timezone

LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
REMOVALS_LOG = os.path.join(LOGS_DIR, "source_removals.jsonl")


def log_removal(source: str, reason: str, actor: str,
                n_chunks: int | None = None, scope: str | None = None,
                extra: dict | None = None) -> dict:
    """Eine Entfernung protokollieren. Gibt den geschriebenen Eintrag zurück."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    entry = {
        "removed_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "reason": reason,
        "actor": actor,
        "n_chunks": n_chunks,
        "scope": scope,
    }
    if extra:
        entry.update(extra)
    with open(REMOVALS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_removals(limit: int | None = None) -> list[dict]:
    """Protokoll einlesen, neueste zuerst."""
    if not os.path.exists(REMOVALS_LOG):
        return []
    out = []
    with open(REMOVALS_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    out.reverse()
    return out[:limit] if limit else out
