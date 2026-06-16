#!/usr/bin/env python3
"""Athena — Coverage-Tracking der Kernfragen (kontinuierliche Verbesserung, messbar).

Misst je Kernfrage, wie gut der RAG sie mit Quellen abdeckt — leichtgewichtig
(nur Retrieval/Embeddings, KEINE LLM-Generierung) → als wöchentlicher Cron tragbar.
Schreibt eine zeitgestempelte Zeile nach eval/coverage_history.jsonl. Steigende
Quellen-Zahl/Relevanz über Zeit = die Wissensbasis wird messbar besser.

Aufruf:  OLLAMA_HOST=… python scripts/coverage_track.py
"""
import json
import os
import sys
import pathlib
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Kernfragen quer über die EVIDENZ-Themen — der Puls der Wissensbasis.
CORE = {
    "rente": "Sollte das gesetzliche Renteneintrittsalter steigen?",
    "buergergeld": "Wie sollte das Bürgergeld reformiert werden?",
    "klima_energie": "Welche Maßnahmen braucht der Klimaschutz im Energiesektor?",
    "migration": "Wie sollte Migration nach Deutschland gesteuert werden?",
    "schuldenbremse": "Sollte die Schuldenbremse für Investitionen reformiert werden?",
    "gesundheit": "Wie sollte die gesetzliche Krankenversicherung reformiert werden?",
    "steuern": "Wie sollte das Steuersystem reformiert werden?",
    "verteidigung": "Wie soll Deutschland seine Verteidigungsausgaben finanzieren?",
    "ki_regulation": "Welche Pflichten gelten für Hochrisiko-KI-Systeme?",
    "wahlrecht": "Welche Wahlrechtsreform ist sinnvoll?",
}


def main():
    import serve
    from retrieval import tier_aware_retrieve
    vs, _ = serve._get_components("bund")
    rows = {}
    for slug, q in CORE.items():
        docs = tier_aware_retrieve(vs, q, k=20, fetch_k=60, sim_floor=0.45, max_k=20)
        srcs = {d.metadata.get("source") for d in docs}
        sims = [d.metadata.get("_similarity") or 0.0 for d in docs]
        rows[slug] = {
            "quellen": len(srcs),
            "chunks": len(docs),
            "avg_relevanz": round(sum(sims) / len(sims), 3) if sims else 0.0,
        }
    rec = {
        "stamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summe_quellen": sum(r["quellen"] for r in rows.values()),
        "topics": rows,
    }
    hist = pathlib.Path(__file__).resolve().parent.parent / "eval" / "coverage_history.jsonl"
    hist.parent.mkdir(parents=True, exist_ok=True)
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[ok] coverage @ {rec['stamp']}: Σ {rec['summe_quellen']} Quellen über {len(rows)} Kernfragen")
    for slug, r in sorted(rows.items(), key=lambda x: -x[1]["quellen"]):
        print(f"   {r['quellen']:3} Quellen · rel {r['avg_relevanz']}  {slug}")


if __name__ == "__main__":
    main()
