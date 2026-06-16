#!/usr/bin/env python3
"""Athena — wiederkehrender Crawl amtlicher Primärquellen-Portale.

Liest Seeds aus config/crawl_seeds.txt, crawlt je Seed neue Dokumente und legt NUR
URLs als Submissions an, die NOCH NICHT bekannt sind — Dedup gegen (a) bereits
ingestierte RAG-Quellen und (b) die Submission-Queue (pending/approved/rejected).
Neue laufen durch auto_review (Tier-0-Vorschlag) und warten auf MENSCHLICHE Freigabe
in der Verify-Queue. So fließen regelmäßig neue amtliche Publikationen rein, ohne
Dubletten und ohne Auto-Ingestion.

Aufruf:  OLLAMA_HOST=… python scripts/recurring_crawl.py [--max-docs 8] [--provider ollama] [--depth 1]
"""
import argparse
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = pathlib.Path(__file__).resolve().parent.parent


def known_urls(scope: str = "bund") -> set:
    """Alle bereits bekannten URLs: ingestierte RAG-Quellen + alle Submissions."""
    urls = set()
    try:
        import serve
        for s in serve._collect_sources(scope):
            u = (s.get("source") or "").rstrip("/")
            if u:
                urls.add(u)
    except Exception as e:
        print(f"  [warn] RAG-Quellen nicht geladen: {e}", file=sys.stderr)
    for m in glob.glob(str(ROOT / "submissions" / "*" / "*" / "meta.json")):
        try:
            d = json.loads(pathlib.Path(m).read_text(encoding="utf-8"))
            u = (d.get("url") or d.get("source") or "").rstrip("/")
            if u:
                urls.add(u)
        except Exception:
            pass
    return urls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-docs", type=int, default=8, help="max. neue Docs je Seed (Kostenbremse)")
    ap.add_argument("--provider", default="ollama", choices=["ollama", "mistral"])
    ap.add_argument("--depth", type=int, default=1)
    a = ap.parse_args()

    from crawl import crawl
    from crawl_ingest import _make_submission, PENDING_DIR
    from auto_review import review_submission

    seeds_file = ROOT / "config" / "crawl_seeds.txt"
    seeds = [l.strip() for l in seeds_file.read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    known = known_urls()
    print(f"[crawl] {len(seeds)} Seeds · {len(known)} bekannte URLs (Dedup)", file=sys.stderr)

    stats = {"neu": 0, "schon_bekannt": 0, "approve": 0, "reject": 0, "needs_human": 0, "error": 0}
    for seed in seeds:
        try:
            res = crawl(seed, a.depth, 60)
        except Exception as e:
            print(f"  [seed-fail] {seed[:55]}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        docs = res.get("documents", [])
        new = [d for d in docs if d["url"].rstrip("/") not in known][:a.max_docs]
        stats["schon_bekannt"] += len(docs) - len(new)
        print(f"  {seed[:55]}: {len(docs)} gefunden, {len(new)} neu", file=sys.stderr)
        for d in new:
            sid = _make_submission(d["url"], "bund", seed)
            known.add(d["url"].rstrip("/"))
            stats["neu"] += 1
            try:
                v = review_submission(PENDING_DIR / sid, provider=a.provider)
                r = v.get("recommendation", "error")
                stats[r] = stats.get(r, 0) + 1
            except Exception as e:
                stats["error"] += 1
                print(f"    [review-fail] {d['url'][:55]}: {type(e).__name__}", file=sys.stderr)
    print(f"[ok] {stats}")
    print("→ neue Vorschläge sind Tier-0 (unverifiziert) — Freigabe in der Verify-Queue.")


if __name__ == "__main__":
    main()
