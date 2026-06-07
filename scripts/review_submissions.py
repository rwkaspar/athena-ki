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
    from retrieval import collection_names_for, get_chroma_client
    scope = meta.get("scope", "pfofeld")
    source_url = meta.get("url") if meta.get("kind") == "url" else None
    client = get_chroma_client()
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
                # 'vorläufig-tierN'-Tag entfernen — die Quelle ist nun echt
                # verifiziert, der vorläufige Rang ist gegenstandslos.
                topics = (m.get("topics") or "")
                if topics:
                    kept = [t for t in topics.split(",")
                            if t.strip() and not t.strip().startswith("vorläufig-tier")]
                    m["topics"] = ",".join(kept)
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
        # kein erzwungenes --render: ingest_url dispatcht PDF/HTML selbst (Render-Fallback).
        args.extend(["--url", meta["url"]])
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


def _uvicorn_active() -> bool:
    try:
        out = subprocess.run(
            ["systemctl", "--user", "is-active", "athena-uvicorn.service"],
            capture_output=True, text=True,
            env={**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"},
        ).stdout.strip()
        return out == "active"
    except FileNotFoundError:
        return False


def _approve(sub_dir: Path, meta: dict, tier: int, label: str) -> str:
    """Eine Submission freigeben (Tier-0 hochstufen oder re-ingestieren)."""
    ar = meta.get("auto_review") or {}
    if ar.get("ingest_status") == "ingested_tier0":
        n = _update_tier0_chunks(meta, tier, label)
        if n == 0:  # nichts hochzustufen → Fallback Re-Ingest
            if ingest_submission(sub_dir, meta, tier, label) != 0:
                return "fail"
    else:
        if ingest_submission(sub_dir, meta, tier, label) != 0:
            return "fail"
    move_to(sub_dir, APPROVED_DIR, extra_meta={
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approved_tier": tier, "approved_label": label, "approved_by": "batch:athena",
    })
    _log_status(meta["id"], "verified", {"verified": True, "tier": tier})
    return "approve"


def _reject(sub_dir: Path, meta: dict, reason: str) -> str:
    if (meta.get("auto_review") or {}).get("ingest_status") == "ingested_tier0":
        _update_tier0_chunks(meta, -1, "")
    move_to(sub_dir, REJECTED_DIR, extra_meta={
        "rejected_at": datetime.now(timezone.utc).isoformat(),
        "reject_reason": reason, "rejected_by": "batch:athena",
    })
    _log_status(meta["id"], "rejected", {"reject_reason": reason})
    return "reject"


def batch_athena(pending, decision: str, dry_run: bool, max_tier: int | None) -> dict:
    """Stapel-Entscheidung nach Athenas Empfehlung. decision = approve|reject."""
    counts = {"matched": 0, "done": 0, "fail": 0, "skipped_tier": 0}
    for sub_dir in pending:
        meta = load_meta(sub_dir)
        ar = meta.get("auto_review") or {}
        if ar.get("recommendation") != decision:
            continue
        counts["matched"] += 1
        if decision == "approve":
            tier = int(ar.get("suggested_tier") or 1)
            if max_tier is not None and tier > max_tier:
                counts["skipped_tier"] += 1
                continue  # unsicherere Tiers manuell lassen
            label = (ar.get("publisher") or meta.get("url") or "Quelle")[:120]
            ident = (meta.get("url") or meta.get("filename") or "")[:70]
            if dry_run:
                print(f"  [approve→T{tier}] {ident}")
                counts["done"] += 1
                continue
            res = _approve(sub_dir, meta, tier, label)
            counts["done" if res == "approve" else "fail"] += 1
            print(f"  {'✓' if res=='approve' else '✗FAIL'} T{tier} | {ident}")
        else:  # reject
            ident = (meta.get("url") or meta.get("filename") or "")[:70]
            if dry_run:
                print(f"  [reject] {ident}")
                counts["done"] += 1
                continue
            _reject(sub_dir, meta, "Athena: " + (ar.get("summary") or "irrelevant")[:120])
            counts["done"] += 1
            print(f"  🗑️  {ident}")
    return counts


def review_pending_missing(pending, provider: str, dry_run: bool, delay: float = 1.5) -> dict:
    """Holt Athenas Auto-Bewertung für Submissions ohne Verdict nach (z.B. nach
    Rate-Limit-Fehlern im Cleanup). Schreibt in ChromaDB (Tier 0) → uvicorn muss aus.
    Throttle (delay) gegen Mistral-Rate-Limit (429). Der Vorfilter in auto_review
    spart bei Müll-Kandidaten den LLM-Call ganz."""
    import time as _t
    from auto_review import review_submission
    counts = {"missing": 0, "reviewed": 0, "prefiltered": 0, "error": 0}
    for sub_dir in pending:
        meta = load_meta(sub_dir)
        if meta.get("auto_review"):
            continue
        counts["missing"] += 1
        ident = (meta.get("url") or meta.get("filename") or "")[:70]
        if dry_run:
            print(f"  [re-review] {ident}")
            continue
        try:
            v = review_submission(sub_dir, provider=provider)
            counts["reviewed"] += 1
            if v.get("prefiltered"):
                counts["prefiltered"] += 1
            else:
                _t.sleep(delay)  # nur nach echtem LLM-Call drosseln
            print(f"  {v.get('recommendation','?'):12} {'(vorfilter)' if v.get('prefiltered') else '':11} | {ident}")
        except Exception as e:
            counts["error"] += 1
            print(f"  ERROR {type(e).__name__}: {str(e)[:40]} | {ident}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Nur Pending-Liste anzeigen")
    parser.add_argument("--id", help="Nur eine bestimmte Submission bearbeiten")
    parser.add_argument("--accept-athena", action="store_true",
                        help="Stapel: alle von Athena 'approve'-empfohlenen freigeben (Tier-Vorschlag)")
    parser.add_argument("--reject-athena", action="store_true",
                        help="Stapel: alle von Athena 'reject'-empfohlenen löschen")
    parser.add_argument("--review-pending", action="store_true",
                        help="Auto-Bewertung für Submissions ohne Verdict nachholen")
    parser.add_argument("--provider", default="mistral", choices=["ollama", "mistral"],
                        help="Provider für --review-pending (Default mistral)")
    parser.add_argument("--review-delay", type=float, default=1.5,
                        help="Sekunden zwischen LLM-Bewertungen bei --review-pending (Rate-Limit)")
    parser.add_argument("--max-tier", type=int, default=None,
                        help="Bei --accept-athena nur Tier<=N automatisch freigeben, Rest manuell")
    parser.add_argument("--dry-run", action="store_true", help="Nur zeigen, nichts ändern")
    parser.add_argument("--allow-uvicorn", action="store_true",
                        help="Schutz übergehen (NICHT empfohlen — Korruptionsrisiko)")
    args = parser.parse_args()

    # SCHUTZ: Freigabe/Reject/Re-Review schreiben direkt in ChromaDB. Im EMBEDDED-
    # Modus droht bei laufendem uvicorn Index-Korruption (Cleanup-Vorfall) → abbrechen.
    # Im SERVER-Modus (chroma_server_mode) serialisiert der Server die Zugriffe →
    # paralleles Reviewen ist sicher, kein Stopp nötig.
    from retrieval import chroma_server_mode
    writes = (args.accept_athena or args.reject_athena or args.review_pending
              or not (args.list or args.dry_run))
    if (writes and not args.dry_run and not args.allow_uvicorn
            and not chroma_server_mode() and _uvicorn_active()):
        sys.exit(
            "[review] ABBRUCH: athena-uvicorn läuft — paralleler ChromaDB-Schreibzugriff "
            "würde den Vektorindex beschädigen.\n"
            "  Erst stoppen:  systemctl --user stop athena-uvicorn.service\n"
            "  Danach Review, dann uvicorn wieder starten.\n"
            "  (Override: --allow-uvicorn)"
        )

    pending = list_pending()
    if not pending:
        print("Keine pending Submissions.")
        return

    if args.review_pending:
        print(f"[review] hole fehlende Auto-Bewertungen nach (provider={args.provider}) …")
        print(review_pending_missing(pending, args.provider, args.dry_run, args.review_delay))
        pending = list_pending()  # Verdicts aktualisiert

    if args.reject_athena:
        print("[review] Stapel-Reject (Athena: reject) …")
        print(batch_athena(pending, "reject", args.dry_run, None))
        pending = list_pending()

    if args.accept_athena:
        print(f"[review] Stapel-Approve (Athena: approve{', Tier<=%d'%args.max_tier if args.max_tier else ''}) …")
        print(batch_athena(pending, "approve", args.dry_run, args.max_tier))
        pending = list_pending()

    if args.accept_athena or args.reject_athena or args.review_pending:
        return  # Stapel-Modus fertig — kein interaktiver Lauf

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
