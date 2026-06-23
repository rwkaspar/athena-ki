#!/usr/bin/env python3
"""Athena — Quellen-Drift-/Vielfalts-Monitoring (Anti-Versteinerung).

Misst NICHT die Abdeckung der Kernfragen (das macht coverage_track.py), sondern die
*Vielfalt* des gesamten Quellen-Korpus: Anzahl Quellen, verschiedene Herausgeber
(Domains), Tier-Verteilung, Themen-Breite und einen normierten Diversitäts-Index
(Shannon-Entropie über die Herausgeber). Hängt eine zeitgestempelte Zeile an
eval/source_drift.jsonl und warnt, wenn sich die Vielfalt gegenüber dem letzten
Lauf verengt — das ist der „institutionelle Versteinerungs"-Frühwarner aus dem
Quellen-Review-Protokoll.

Aufruf:  OLLAMA_HOST=… python scripts/source_drift.py [--scope bund]
"""
import argparse
import json
import math
import os
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HIST = pathlib.Path(__file__).resolve().parent.parent / "eval" / "source_drift.jsonl"


def _publisher(src: str) -> str:
    """Herausgeber = Domain (ohne www). Nicht-URLs (Datei-Uploads) als eigener Bucket."""
    s = (src or "").strip()
    if s.startswith("http"):
        net = urlparse(s).netloc.lower()
        return net[4:] if net.startswith("www.") else net or "(unbekannt)"
    return "(datei/sonstige)"


def _norm_entropy(counts) -> float:
    """Normierte Shannon-Entropie (0=monokultur, 1=maximal vielfältig)."""
    n = sum(counts)
    if n <= 0 or len(counts) <= 1:
        return 0.0
    h = -sum((c / n) * math.log(c / n) for c in counts if c)
    return round(h / math.log(len(counts)), 3)


def _last_record():
    if not HIST.exists():
        return None
    last = None
    for line in HIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                last = json.loads(line)
            except Exception:
                pass
    return last


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scope", default="bund")
    a = ap.parse_args()

    import serve
    sources = serve._collect_sources(a.scope)

    publishers = Counter(_publisher(s.get("source")) for s in sources)
    tiers = Counter(str(s.get("tier_rank")) for s in sources)
    topics = Counter()
    for s in sources:
        for t in (s.get("topics") or []):
            topics[t] += 1

    n = len(sources)
    top_pub, top_n = publishers.most_common(1)[0] if publishers else ("—", 0)
    rec = {
        "stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scope": a.scope,
        "n_sources": n,
        "n_publishers": len(publishers),
        "n_topics": len(topics),
        "diversity": _norm_entropy(list(publishers.values())),   # 0..1 über Herausgeber
        "top_publisher_share": round(top_n / n, 3) if n else 0.0,
        "tier_hist": dict(sorted(tiers.items())),
    }

    prev = _last_record()
    HIST.parent.mkdir(parents=True, exist_ok=True)
    with open(HIST, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ok] drift @ {rec['stamp']}  scope={a.scope}")
    print(f"   {n} Quellen · {rec['n_publishers']} Herausgeber · {rec['n_topics']} Themen")
    print(f"   Diversität {rec['diversity']} · größter Herausgeber {top_pub} ({rec['top_publisher_share']*100:.0f}%)")
    print(f"   Tier-Verteilung {rec['tier_hist']}")

    if prev:
        warns = []
        if rec["diversity"] < prev.get("diversity", 0) - 0.02:
            warns.append(f"Diversität sinkt {prev['diversity']} → {rec['diversity']}")
        if rec["n_publishers"] < prev.get("n_publishers", 0):
            warns.append(f"weniger Herausgeber {prev['n_publishers']} → {rec['n_publishers']}")
        if rec["top_publisher_share"] > prev.get("top_publisher_share", 1) + 0.05:
            warns.append(f"Konzentration steigt {prev['top_publisher_share']} → {rec['top_publisher_share']}")
        for t in ("0", "1", "2", "3"):
            if rec["tier_hist"].get(t, 0) < prev.get("tier_hist", {}).get(t, 0):
                warns.append(f"Tier {t}: {prev['tier_hist'].get(t,0)} → {rec['tier_hist'].get(t,0)}")
        if warns:
            print("   ⚠ DRIFT-WARNUNG (Verengung):")
            for w in warns:
                print(f"      · {w}")
        else:
            print("   ✓ keine Verengung gegenüber letztem Lauf")


if __name__ == "__main__":
    main()
