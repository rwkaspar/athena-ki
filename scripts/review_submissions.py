#!/usr/bin/env python3
"""Athena - Manueller Review-CLI für eingereichte Quellen.

Geht durch alle Submissions in submissions/pending/, zeigt jede an, fragt
approve (mit Tier-Wahl) oder reject. Approvals werden via ingest.py direkt
in die ChromaDB aufgenommen und in submissions/approved/ verschoben.
Rejections wandern mit Grund nach submissions/rejected/.

Verwendung:
    python scripts/review_submissions.py                # Default-Workflow
    python scripts/review_submissions.py --list         # nur Liste
    python scripts/review_submissions.py --id <hex>     # nur einen Eintrag
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
PENDING_DIR = SUBMISSIONS_DIR / "pending"
APPROVED_DIR = SUBMISSIONS_DIR / "approved"
REJECTED_DIR = SUBMISSIONS_DIR / "rejected"
INGEST_SCRIPT = Path(__file__).parent / "ingest.py"


def list_pending() -> list[Path]:
    if not PENDING_DIR.exists():
        return []
    return sorted(
        (p for p in PENDING_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists()),
        key=lambda p: (p / "meta.json").stat().st_mtime,
    )


def load_meta(submission_dir: Path) -> dict:
    return json.loads((submission_dir / "meta.json").read_text(encoding="utf-8"))


CHROMA_DB_DIR = Path(__file__).parent.parent / "athena-db"


def _source_value(meta: dict) -> str:
    """Der 'source'-Metadatenwert, unter dem die Chunks in ChromaDB liegen."""
    return meta["url"] if meta.get("kind") == "url" else str(
        (Path(meta["_dir"]) / meta["filename"])) if meta.get("_dir") else meta.get("filename", "")


def _update_tier0_chunks(meta: dict, new_tier: int, label: str) -> int:
    """Tier-0-Chunks dieser Quelle auf new_tier hochstufen (verifizieren) ODER
    bei new_tier<0 löschen. Operiert direkt auf ChromaDB. Liefert Anzahl Chunks.

    Eingereichte Quellen sind nach auto_review bereits als Tier 0 in der DB —
    die Freigabe ändert nur das tier_rank-Metadatum, kein Re-Ingest nötig."""
    import chromadb
    from retrieval import collection_names_for
    scope = meta.get("scope", "pfofeld")
    source_url = meta.get("url") if meta.get("kind") == "url" else None
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    affected = 0
    for coll_name in collection_names_for(scope):
        try:
            coll = client.get_collection(coll_name)
        except Exception:
            continue
        # Chunks dieser Quelle mit tier_rank 0 finden
        where = {"$and": [{"source": source_url}, {"tier_rank": 0}]} if source_url else {"tier_rank": 0}
        try:
            got = coll.get(where=where)
        except Exception:
            continue
        ids = got.get("ids") or []
        if not ids:
            continue
        if new_tier < 0:
            coll.delete(ids=ids)
            affected += len(ids)
        else:
            metas = got.get("metadatas") or []
            for m in metas:
                m["tier_rank"] = new_tier
                m["tier_label"] = label
                m["source_type"] = "static" if new_tier == 1 else "fresh"
                m["verified_at"] = datetime.now(timezone.utc).isoformat()
            coll.update(ids=ids, metadatas=metas)
            affected += len(ids)
    return affected


def _log_status(submission_id: str, new_status: str, extra: dict | None = None):
    """Status im öffentlichen Prüf-Protokoll (log.jsonl) nachführen — hängt eine
    Statuszeile an, damit die Seite den finalen Zustand (verified/rejected) zeigt."""
    log_path = SUBMISSIONS_DIR / "log.jsonl"
    entry = {"id": submission_id, "status": new_status,
             "updated_at": datetime.now(timezone.utc).isoformat()}
    if extra:
        entry.update(extra)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def print_summary(idx: int, total: int, submission_dir: Path, meta: dict) -> None:
    print("\n" + "=" * 70)
    print(f"[{idx}/{total}]  ID: {meta.get('id')}  ·  eingereicht: {meta.get('submitted_at')}")
    print(f"  Scope: {meta.get('scope', 'pfofeld')}")
    print(f"  Kind: {meta.get('kind')}")
    if meta.get("kind") == "url":
        print(f"  URL:  {meta.get('url')}")
    else:
        print(f"  File: {meta.get('filename')}  ({meta.get('size_bytes')} bytes, {meta.get('content_type')})")
    if meta.get("note"):
        print(f"  Notiz: {meta['note'][:300]}")
    print(f"  Quelle (IP): {meta.get('source_ip')}  ·  UA: {(meta.get('user_agent') or '')[:80]}")
    # Athenas automatische Vorab-Bewertung als Entscheidungshilfe
    ar = meta.get("auto_review")
    if ar:
        print("  " + "-" * 66)
        print(f"  🤖 ATHENA: {ar.get('recommendation','?').upper()}  ·  {ar.get('summary','')}")
        print(f"     Herausgeber: {ar.get('publisher','?')}  (Vertrauen: {ar.get('publisher_trust','?')})")
        print(f"     Relevant: {ar.get('relevant')}  ·  Tier-Vorschlag: {ar.get('suggested_tier')}  ·  Status: {ar.get('ingest_status','-')}")
        print(f"     Tags: {', '.join(ar.get('topics') or []) or '—'}")
    print(f"  Dir: {submission_dir}")
    print("=" * 70)


def ingest_submission(submission_dir: Path, meta: dict, tier: int, label: str) -> int:
    """Ruft ingest.py mit den passenden Parametern auf, vererbt env. Liefert rc."""
    scope = meta.get("scope", "pfofeld")
    args = [
        sys.executable, str(INGEST_SCRIPT),
        "--tier", str(tier),
        "--source-label", label,
        "--scope", scope,
    ]
    if meta["kind"] == "url":
        args.extend(["--url", meta["url"], "--render"])
    else:
        file_path = submission_dir / meta["filename"]
        args.extend(["--file", str(file_path)])
    print(f"\n   ➜ Ingest: {' '.join(args)}")
    rc = subprocess.call(args)
    return rc


def move_to(submission_dir: Path, target_dir: Path, extra_meta: dict | None = None) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    if extra_meta:
        meta_path = submission_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(extra_meta)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    dest = target_dir / submission_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(submission_dir), str(dest))
    return dest


def prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    ans = input(f"{text}{suffix}: ").strip()
    return ans or (default or "")


def review_one(submission_dir: Path) -> str:
    meta = load_meta(submission_dir)
    ar = meta.get("auto_review") or {}
    already_tier0 = (ar.get("ingest_status") == "ingested_tier0")
    while True:
        action = prompt("Aktion ([a]pprove / [r]eject / [s]kip / [q]uit)", "s").lower()
        if action in ("a", "approve"):
            default_tier = str(ar.get("suggested_tier") or 1)
            tier_str = prompt("Tier (1=Primär, 2=Medien, 3=Kommentar)", default_tier)
            if tier_str not in ("1", "2", "3"):
                print("  ❌ ungültiges Tier")
                continue
            label = prompt("Source-Label", (ar.get("publisher") or "User-Submission")[:120])
            if already_tier0:
                # Quelle ist schon als Tier 0 in ChromaDB → nur hochstufen (verifizieren)
                n = _update_tier0_chunks(meta, int(tier_str), label)
                if n == 0:
                    print("  ⚠️ Keine Tier-0-Chunks gefunden — fällt auf Re-Ingest zurück.")
                    rc = ingest_submission(submission_dir, meta, int(tier_str), label)
                    if rc != 0:
                        print(f"  ⚠️ Ingest fehlgeschlagen (rc={rc})"); return "fail"
                else:
                    print(f"  ✓ {n} Chunks von Tier 0 → Tier {tier_str} hochgestuft (verifiziert)")
            else:
                rc = ingest_submission(submission_dir, meta, int(tier_str), label)
                if rc != 0:
                    print(f"  ⚠️ Ingest fehlgeschlagen (rc={rc}), Submission bleibt pending")
                    return "fail"
            move_to(submission_dir, APPROVED_DIR, extra_meta={
                "approved_at": datetime.now(timezone.utc).isoformat(),
                "approved_tier": int(tier_str),
                "approved_label": label,
            })
            _log_status(meta["id"], "verified", {"verified": True, "tier": int(tier_str)})
            print(f"  ✓ freigegeben (Tier {tier_str}) und nach approved/ verschoben")
            return "approve"
        elif action in ("r", "reject"):
            reason = prompt("Grund (optional)", "irrelevant / unseriös / kein Bezug")
            if already_tier0:
                # Tier-0-Chunks aus ChromaDB entfernen (waren nur unverifiziert drin)
                n = _update_tier0_chunks(meta, -1, "")
                print(f"  🗑️  {n} Tier-0-Chunks aus der Wissensbasis gelöscht")
            move_to(submission_dir, REJECTED_DIR, extra_meta={
                "rejected_at": datetime.now(timezone.utc).isoformat(),
                "reject_reason": reason,
            })
            _log_status(meta["id"], "rejected", {"reject_reason": reason})
            print(f"  ✗ nach rejected/ verschoben")
            return "reject"
        elif action in ("s", "skip", ""):
            return "skip"
        elif action in ("q", "quit"):
            return "quit"
        else:
            print(f"  ? '{action}' nicht verstanden")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Nur Pending-Liste anzeigen")
    parser.add_argument("--id", help="Nur eine bestimmte Submission bearbeiten")
    args = parser.parse_args()

    pending = list_pending()
    if not pending:
        print("Keine pending Submissions.")
        return

    if args.list:
        print(f"{len(pending)} pending Submission(s):\n")
        for p in pending:
            meta = load_meta(p)
            kind = meta.get("kind", "?")
            ident = meta.get("url") if kind == "url" else meta.get("filename")
            print(f"  {meta['id']}  [{kind}]  {meta.get('submitted_at', '?')[:19]}  {ident}")
        return

    if args.id:
        pending = [p for p in pending if p.name.startswith(args.id) or load_meta(p).get("id", "").startswith(args.id)]
        if not pending:
            print(f"Keine Submission mit ID {args.id}")
            return

    counts = {"approve": 0, "reject": 0, "skip": 0, "fail": 0}
    for i, sub_dir in enumerate(pending, 1):
        meta = load_meta(sub_dir)
        print_summary(i, len(pending), sub_dir, meta)
        result = review_one(sub_dir)
        counts[result] = counts.get(result, 0) + 1
        if result == "quit":
            break

    print(f"\nFertig. {counts}")


if __name__ == "__main__":
    main()
