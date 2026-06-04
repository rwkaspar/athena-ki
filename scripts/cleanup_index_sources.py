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
import time
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

# sieht nach echtem Inhaltsdokument aus (PDF oder Detail-Slug einer Publikation)
REAL_DOC_RE = re.compile(
    r"\.pdf($|\?)|/(publikation|dokument|gutachten|bericht|studie|stellungnahme|"
    r"drucksache|entscheidung|artikel|dp)/?[\w-]{4,}|/fileadmin/.+\.pdf",
    re.I,
)
# klar nicht-inhaltliche Slugs (Org/Navigation), die trotz Prosa irrelevant sind
NAV_SLUG_RE = re.compile(
    r"/(team|ueber|das-zew|kontakt|veranstaltungen|leichte-sprache|gebaerdensprache|"
    r"karriere|jobs|newsletter|forschungsschwerpunkte|team-kontakt|organigramm|"
    r"ueber-uns|ansprechpartner)\b",
    re.I,
)


def rank_documents(docs: list[dict]) -> list[dict]:
    """Echte Dokumente (PDF/Detail-Slug) zuerst, klare Navigations-Slugs ans Ende.
    Der Crawler liefert in BFS-Reihenfolge — sonst landen Navi-Seiten im max_docs-Fenster
    und die eigentlichen Studien (oft tiefer) fallen raus."""
    def score(d):
        u = d.get("url", "")
        if NAV_SLUG_RE.search(u):
            return 2          # zuletzt
        if REAL_DOC_RE.search(u):
            return 0          # zuerst
        return 1
    return sorted(docs, key=score)


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
    p.add_argument("--review-delay", type=float, default=1.0, help="Sekunden Pause zwischen Bewertungen (gegen Rate-Limit)")
    p.add_argument("--dry-run", action="store_true", help="nur crawlen + zeigen, nichts bewerten/löschen")
    p.add_argument("--seeds-file", help="JSON-Liste von Seed-URLs (sonst aus ChromaDB ermittelt)")
    p.add_argument("--allow-uvicorn", action="store_true",
                   help="Schutz übergehen und trotz laufendem uvicorn ingestieren (NICHT empfohlen)")
    args = p.parse_args()

    # SCHUTZ gegen DB-Korruption: ChromaDB verträgt keinen parallelen Multi-Prozess-
    # Schreibzugriff. Läuft uvicorn (hält dieselbe DB offen), würde der Bulk-Ingest
    # den hnswlib-Index zerstören (siehe Vorfall bund_fresh). Daher abbrechen.
    if not args.dry_run and not args.allow_uvicorn:
        import subprocess
        try:
            active = subprocess.run(
                ["systemctl", "--user", "is-active", "athena-uvicorn.service"],
                capture_output=True, text=True,
                env={**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"},
            ).stdout.strip()
        except FileNotFoundError:
            active = "unknown"
        if active == "active":
            sys.exit(
                "[cleanup] ABBRUCH: athena-uvicorn läuft — paralleler ChromaDB-Schreibzugriff "
                "würde den Vektorindex beschädigen.\n"
                "  Erst stoppen:   systemctl --user stop athena-uvicorn.service\n"
                "  Danach Cleanup laufen lassen, dann uvicorn wieder starten.\n"
                "  (Override nur wenn du weißt was du tust: --allow-uvicorn)"
            )

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
        docs = rank_documents(res["documents"])[:args.max_docs]
        grand["docs_found"] += len(docs)
        print(f"  {len(res['documents'])} Dokumente (priorisiert), bewerte {len(docs)}", file=sys.stderr)
        if args.dry_run:
            for d in docs:
                print(f"    [{d['type']}] {d['url']}")
            continue
        harvested = 0  # erfolgreich bewertete Dokumente (egal welche Empfehlung)
        for d in docs:
            sub_id = make_submission(d["url"], args.scope, seed)
            try:
                v = review_submission(PENDING_DIR / sub_id, provider=args.provider)
                rec = v.get("recommendation", "error")
                grand[rec] = grand.get(rec, 0) + 1
                harvested += 1
                print(f"    {rec:12} | {d['url'][:65]}", file=sys.stderr)
            except Exception as e:
                grand["error"] += 1
                print(f"    ERROR {type(e).__name__}: {str(e)[:50]} | {d['url'][:45]}", file=sys.stderr)
            if args.review_delay:
                time.sleep(args.review_delay)  # Throttle gegen Mistral-Rate-Limit (429)
        # SCHUTZ: Index-Seite NUR löschen, wenn mindestens ein Dokument erfolgreich
        # geerntet wurde — sonst gingen Quellen verloren, ohne Ersatz zu schaffen.
        if harvested == 0:
            print(f"  ⏭️  Index-Seite NICHT gelöscht (0 Dokumente geerntet — alle Fehler)", file=sys.stderr)
            continue
        n = delete_source(args.scope, seed)
        grand["deleted_chunks"] += n
        grand["seeds"] += 1
        print(f"  🗑️  Index-Seite gelöscht ({n} chunks, {harvested} Dokumente geerntet)", file=sys.stderr)

    print(f"\nFertig. {json.dumps(grand, ensure_ascii=False)}")
    if not args.dry_run:
        print("→ Geerntete Dokumente sind Tier-0 (unverifiziert). Freigabe: review_submissions.py")


if __name__ == "__main__":
    main()
