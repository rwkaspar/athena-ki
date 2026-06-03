#!/usr/bin/env python3
"""Athena — Index-Seiten-Cleanup: ernten statt wegwerfen.

Für jede Index-/Übersichtsseite in der Wissensbasis:
  1. crawlen → echte Dokument-URLs finden (crawl.py)
  2. jedes Dokument als Tier-0-Vorschlag bewerten (auto_review, ollama)
  3. die wertlose Index-Seite aus ChromaDB löschen

So wird aus einer Linkliste eine kuratierte Quellenmenge, ohne Müll zu behalten.
Mensch verifiziert die Tier-0-Vorschläge später (review_submissions.py).

Aufruf:
    python scripts/cleanup_index_sources.py --scope bund [--limit 3]
        [--max-docs 8] [--dry-run] [--seeds-file /tmp/index_seeds.json]
"""

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chromadb
from crawl import crawl
from auto_review import review_submission
from retrieval import collection_names_for

CHROMA_DB_DIR = Path(__file__).parent.parent / "athena-db"
SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
PENDING_DIR = SUBMISSIONS_DIR / "pending"

# erkennt Index-/Übersichts-URLs
INDEX_RE = re.compile(
    r"/publikationen/?$|/themen(/|_node\.html|/_inhalt\.html)?$|/aktuelles/?$"
    r"|/veroeffentlichungen/?$|/publikation/?$|/meldungen/?$|_node\.html$|/themen\.html$",
    re.I,
)


def find_index_sources(scope: str) -> list[str]:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    urls = set()
    for coll_name in collection_names_for(scope):
        try:
            coll = client.get_collection(coll_name)
        except Exception:
            continue
        for m in (coll.get().get("metadatas") or []):
            u = (m or {}).get("source") or ""
            if u.startswith("http") and INDEX_RE.search(u):
                urls.add(u)
    return sorted(urls)


def delete_source(scope: str, url: str) -> int:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    n = 0
    for coll_name in collection_names_for(scope):
        try:
            coll = client.get_collection(coll_name)
        except Exception:
            continue
        got = coll.get(where={"source": url})
        ids = got.get("ids") or []
        if ids:
            coll.delete(ids=ids)
            n += len(ids)
    return n


def make_submission(url: str, scope: str, seed: str) -> str:
    sub_id = uuid.uuid4().hex[:12]
    target = PENDING_DIR / sub_id
    target.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": sub_id, "submitted_at": datetime.now(timezone.utc).isoformat(),
        "kind": "url", "scope": scope, "url": url,
        "note": f"Geerntet aus Index-Seite {seed}",
        "origin": "cleanup_crawl", "seed": seed, "user_agent": "cleanup",
    }
    (target / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return sub_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", default="bund", choices=["pfofeld", "bund"])
    p.add_argument("--limit", type=int, default=3, help="max. Index-Seiten in diesem Lauf")
    p.add_argument("--max-docs", type=int, default=8, help="max. Dokumente pro Index-Seite")
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--provider", default="ollama", choices=["ollama", "mistral"])
    p.add_argument("--dry-run", action="store_true", help="nur crawlen + zeigen, nichts bewerten/löschen")
    p.add_argument("--seeds-file", help="JSON-Liste von Seed-URLs (sonst aus ChromaDB ermittelt)")
    args = p.parse_args()

    if args.seeds_file and Path(args.seeds_file).exists():
        seeds = json.loads(Path(args.seeds_file).read_text())
    else:
        seeds = find_index_sources(args.scope)
    seeds = seeds[:args.limit]
    print(f"[cleanup] {len(seeds)} Index-Seiten in diesem Lauf (scope={args.scope})", file=sys.stderr)

    grand = {"docs_found": 0, "approve": 0, "reject": 0, "needs_human": 0, "error": 0, "deleted_chunks": 0, "seeds": 0}
    for si, seed in enumerate(seeds, 1):
        print(f"\n[{si}/{len(seeds)}] Crawle {seed}", file=sys.stderr)
        try:
            res = crawl(seed, args.depth, max_pages=25)
        except Exception as e:
            print(f"  Crawl-Fehler: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        docs = res["documents"][:args.max_docs]
        grand["docs_found"] += len(docs)
        print(f"  {len(res['documents'])} Dokumente, bewerte {len(docs)}", file=sys.stderr)
        if args.dry_run:
            for d in docs:
                print(f"    [{d['type']}] {d['url']}")
            continue
        for d in docs:
            sub_id = make_submission(d["url"], args.scope, seed)
            try:
                v = review_submission(PENDING_DIR / sub_id, provider=args.provider)
                rec = v.get("recommendation", "error")
                grand[rec] = grand.get(rec, 0) + 1
                print(f"    {rec:12} | {d['url'][:65]}", file=sys.stderr)
            except Exception as e:
                grand["error"] += 1
                print(f"    ERROR {type(e).__name__} | {d['url'][:55]}", file=sys.stderr)
        # Index-Seite löschen (ihre Dokumente sind jetzt geerntet)
        n = delete_source(args.scope, seed)
        grand["deleted_chunks"] += n
        grand["seeds"] += 1
        print(f"  🗑️  Index-Seite gelöscht ({n} chunks)", file=sys.stderr)

    print(f"\nFertig. {json.dumps(grand, ensure_ascii=False)}")
    if not args.dry_run:
        print("→ Geerntete Dokumente sind Tier-0 (unverifiziert). Freigabe: review_submissions.py")


if __name__ == "__main__":
    main()
