#!/usr/bin/env python3
"""Eine ECHTE Quelle aus der Wissensbasis entfernen — MIT Begründung + Audit-Log.

Das ist der vorgeschriebene Weg statt ad-hoc-Löschen: jede Entfernung wird
nachvollziehbar in logs/source_removals.jsonl protokolliert und ist über
/source-removals einsehbar.

  # Vorschau:
  athena-env/bin/python scripts/remove_source.py --source <url> --reason "..."
  # Anwenden (uvicorn vorher stoppen):
  systemctl --user stop athena-uvicorn
  athena-env/bin/python scripts/remove_source.py --source <url> --reason "..." --apply
  systemctl --user start athena-uvicorn

Leeren Müll/Orphans NICHT hierüber löschen (die brauchen kein Audit).
"""
import argparse
import os
import subprocess
import sys

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import chromadb
from retrieval import get_chroma_client, chroma_server_mode
from source_audit import log_removal

DB = os.path.join(os.path.dirname(__file__), "..", "athena-db")
SCOPE_COLLECTIONS = {"bund": ["bund_static", "bund_fresh"], "pfofeld": ["static", "fresh"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="exakte source-URL/Pfad")
    ap.add_argument("--reason", required=True, help="Begründung (wird protokolliert)")
    ap.add_argument("--scope", default="bund", choices=list(SCOPE_COLLECTIONS))
    ap.add_argument("--actor", default="manual", help="wer/was entfernt (z.B. Name)")
    ap.add_argument("--apply", action="store_true", help="wirklich löschen (sonst dry-run)")
    ap.add_argument("--force", action="store_true", help="auch bei aktivem uvicorn löschen")
    a = ap.parse_args()

    client = get_chroma_client()
    per = {}
    total = 0
    for cn in SCOPE_COLLECTIONS[a.scope]:
        try:
            coll = client.get_collection(cn)
        except Exception:
            continue
        ids = coll.get(where={"source": a.source}).get("ids") or []
        per[cn] = ids
        total += len(ids)

    print(f"Quelle : {a.source}")
    print(f"Scope  : {a.scope}")
    print(f"Chunks : {total}  ({', '.join(f'{k}:{len(v)}' for k, v in per.items())})")
    print(f"Grund  : {a.reason}")
    if total == 0:
        print("\nNichts zu löschen (Quelle nicht gefunden).")
        return
    if not a.apply:
        print("\n(DRY-RUN — nichts gelöscht. Mit --apply ausführen, uvicorn vorher stoppen.)")
        return

    # Im Server-Modus serialisiert der Chroma-Server die Zugriffe → kein Stopp nötig.
    if not chroma_server_mode():
        active = subprocess.run(["systemctl", "--user", "is-active", "athena-uvicorn"],
                                capture_output=True, text=True).stdout.strip()
        if active == "active" and not a.force:
            print("\n[ABBRUCH] athena-uvicorn ist AKTIV — erst stoppen "
                  "(systemctl --user stop athena-uvicorn) oder --force.", file=sys.stderr)
            sys.exit(1)

    for cn, ids in per.items():
        if ids:
            client.get_collection(cn).delete(ids=ids)
    entry = log_removal(a.source, a.reason, a.actor, n_chunks=total, scope=a.scope)
    print(f"\n[ok] {total} Chunks gelöscht + protokolliert ({entry['removed_at']}).")
    print("Einsehbar unter /source-removals.")


if __name__ == "__main__":
    main()
