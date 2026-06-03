#!/usr/bin/env python3
"""Athena — Crawl → Auto-Review-Orchestrierung.

Nimmt eine Index-/Übersichtsseite, crawlt die echten Dokumente (crawl.py),
legt für jedes eine Submission an und lässt Athena sie bewerten (auto_review).
Akzeptierte landen als Tier-0-Vorschläge — die finale Freigabe macht ein Mensch
(review_submissions.py). So wird aus einer wertlosen Index-Seite eine kuratierte,
verifizierbare Quellenliste.

Empfohlen mit --provider ollama (viele Dokumente → kein Mistral-Rate-Limit).

Aufruf:
    python scripts/crawl_ingest.py <seed-url> --scope bund [--depth 2]
        [--max-pages 60] [--max-docs 20] [--provider ollama] [--dry-run]
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crawl import crawl
from auto_review import review_submission

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
PENDING_DIR = SUBMISSIONS_DIR / "pending"


def _make_submission(url: str, scope: str, seed: str) -> str:
    """Legt eine pending-Submission für eine gecrawlte URL an. Liefert die id."""
    sub_id = uuid.uuid4().hex[:12]
    target = PENDING_DIR / sub_id
    target.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": sub_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "kind": "url",
        "scope": scope,
        "url": url,
        "note": f"Auto-gecrawlt von {seed}",
        "source_ip": None,
        "user_agent": "crawl_ingest",
        "origin": "crawl",
        "seed": seed,
    }
    (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return sub_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", help="Index-/Übersichts-URL")
    p.add_argument("--scope", default="bund", choices=["pfofeld", "bund"])
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--max-pages", type=int, default=60)
    p.add_argument("--max-docs", type=int, default=20, help="max. Dokumente zur Bewertung (Kostenbremse)")
    p.add_argument("--provider", default="ollama", choices=["ollama", "mistral"])
    p.add_argument("--dry-run", action="store_true", help="nur crawlen + Doc-Liste zeigen, nicht bewerten/anlegen")
    args = p.parse_args()

    print(f"[crawl_ingest] Crawle {args.seed} …", file=sys.stderr)
    result = crawl(args.seed, args.depth, args.max_pages)
    docs = result["documents"][:args.max_docs]
    print(f"[crawl_ingest] {len(result['documents'])} Dokumente gefunden, "
          f"bewerte {len(docs)} (max-docs={args.max_docs})", file=sys.stderr)

    if args.dry_run:
        for d in docs:
            print(f"  [{d['type']}] {d['url']}")
        print(f"\nDRY-RUN: {len(docs)} Dokumente würden als Tier-0-Vorschläge bewertet.")
        return

    stats = {"approve": 0, "reject": 0, "needs_human": 0, "error": 0}
    for i, d in enumerate(docs, 1):
        sub_id = _make_submission(d["url"], args.scope, args.seed)
        try:
            verdict = review_submission(PENDING_DIR / sub_id, provider=args.provider)
            rec = verdict.get("recommendation", "error")
            stats[rec] = stats.get(rec, 0) + 1
            print(f"  [{i}/{len(docs)}] {rec:12} | {d['url'][:70]}", file=sys.stderr)
        except Exception as e:
            stats["error"] += 1
            print(f"  [{i}/{len(docs)}] ERROR {type(e).__name__}: {d['url'][:60]}", file=sys.stderr)

    print(f"\nFertig. {stats}")
    print("→ Vorschläge sind Tier-0 (unverifiziert). Freigabe: scripts/review_submissions.py")


if __name__ == "__main__":
    main()
