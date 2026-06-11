#!/usr/bin/env python3
"""Athena — tote Links aus der Pending-Queue aussortieren.

Viele auto-gecrawlte Submissions zeigen auf inzwischen tote URLs (404/410) —
z. B. crawler-verbogene destatis-Pfade. Diese sollen den Menschen im Review
nicht belasten. Dieses Skript prüft jede Pending-URL per HTTP und:

  - HTTP 404/410            -> automatisch nach rejected/ (eindeutig tot)
  - 200/2xx/3xx            -> bleibt pending (lebt)
  - 403/401/400/5xx/Timeout -> bleibt pending, wird nur GEMELDET
                              (kann UA-Block oder vorübergehend sein)

Aufruf:
    python scripts/prune_dead_pending.py            # echte Bereinigung
    python scripts/prune_dead_pending.py --dry-run  # nur zeigen
    python scripts/prune_dead_pending.py --also-400 # 400 ebenfalls als tot werten
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_submissions import PENDING_DIR, REJECTED_DIR, load_meta, move_to, _log_status

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
DEAD_CODES = {404, 410}


def check(url: str) -> int:
    """HTTP-Status der URL; -1 = Netzwerkfehler/Timeout. Erst HEAD, dann GET-Fallback."""
    if not url:
        return -2  # keine URL
    for method in ("head", "get"):
        try:
            r = requests.request(method, url, timeout=12, allow_redirects=True,
                                 headers={"User-Agent": UA}, stream=(method == "get"))
            code = r.status_code
            r.close()
            # manche Server beantworten HEAD mit 405/403 — dann GET nachschieben
            if method == "head" and code in (403, 405, 400):
                continue
            return code
        except Exception:
            if method == "get":
                return -1
    return -1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--also-400", action="store_true", help="HTTP 400 ebenfalls als tot werten")
    args = ap.parse_args()
    dead_codes = DEAD_CODES | ({400} if args.also_400 else set())

    subs = [p for p in PENDING_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists()]
    items = []
    for sd in subs:
        m = load_meta(sd)
        items.append((sd, m.get("url") or m.get("source") or ""))
    print(f"[prune] prüfe {len(items)} Pending-URLs …", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=16) as ex:
        codes = list(ex.map(lambda it: check(it[1]), items))

    dead, alive, doubt = [], [], []
    for (sd, url), code in zip(items, codes):
        if code in dead_codes:
            dead.append((sd, url, code))
        elif 200 <= code < 400:
            alive.append((sd, url, code))
        else:
            doubt.append((sd, url, code))

    # Domains-Übersicht der toten
    from collections import Counter
    from urllib.parse import urlparse
    dom = Counter(urlparse(u).netloc for _, u, _ in dead)

    print(f"\n=== Ergebnis: {len(alive)} lebendig · {len(dead)} tot · {len(doubt)} unklar ===")
    print("Tote Links nach Domain:")
    for d, n in dom.most_common():
        print(f"  {n:4}  {d}")
    if doubt:
        dc = Counter(c for _, _, c in doubt)
        print("Unklar (bleiben pending) nach Code:", dict(dc))

    if args.dry_run:
        print("\nDRY-RUN — nichts verschoben.")
        return

    now = datetime.now(timezone.utc).isoformat()
    for sd, url, code in dead:
        sid = sd.name
        move_to(sd, REJECTED_DIR, extra_meta={
            "rejected_at": now,
            "reject_reason": f"Toter Link (HTTP {code}) — automatisch aussortiert (prune_dead_pending).",
            "decided_by": "auto-prune"})
        _log_status(sid, "rejected", {"reject_reason": f"dead link HTTP {code}",
                                      "decided_via": "auto-prune"})
    print(f"\n{len(dead)} tote Submissions → rejected/. {len(alive)} bleiben pending, "
          f"{len(doubt)} unklar bleiben pending.")


if __name__ == "__main__":
    main()
