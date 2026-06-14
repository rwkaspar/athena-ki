#!/usr/bin/env python3
"""Athena — RAG-Transparenz-Demo: macht das „offene Buch" sichtbar.

Für eine Frage zeigt es GENAU, welche Quellen-Ausschnitte (Chunks) das Modell
vorgelegt bekommt — dieselbe Pipeline wie der Live-Chat (tier_aware_retrieve,
k=5, fetch_k=20, Tier-Boost, Sprach-Filter). Das Modell „kennt" diese Daten
nicht aus dem Training, es liest sie zur Laufzeit.

Ausgabe: Markdown nach stdout (oder --out FILE) + optional JSON (--json FILE),
damit der Website-Build es als Transparenz-Sektion rendern kann.

Aufruf:
    OLLAMA_HOST=http://… python scripts/rag_demo.py "Frage…" [--scope bund]
        [--out demo.md] [--json demo.json]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run(question: str, scope: str = "bund", excerpt: int = 280):
    import serve
    from retrieval import tier_aware_retrieve
    vs, _ = serve._get_components(scope)
    docs = tier_aware_retrieve(vs, question, k=serve.RETRIEVER_K, fetch_k=serve.RETRIEVER_FETCH_K)
    out = []
    for d in docs:
        m = d.metadata
        out.append({
            "tier": m.get("tier_rank"),
            "score": m.get("_combined_score"),
            "similarity": m.get("_similarity"),
            "title": m.get("title") or m.get("source"),
            "source": m.get("source"),
            "excerpt": " ".join((d.page_content or "").split())[:excerpt],
        })
    return {"question": question, "scope": scope,
            "k": serve.RETRIEVER_K, "fetch_k": serve.RETRIEVER_FETCH_K,
            "chunks": out}


def to_markdown(r: dict) -> str:
    L = ['# Was Athena „vorgelegt“ bekommt', "",
         f"**Frage:** {r['question']}", "",
         f"Aus der Wissensbasis (Scope *{r['scope']}*) werden die {r['fetch_k']} "
         f"ähnlichsten Ausschnitte gesucht, nach Tier (Quellengüte) neu gewichtet, "
         f"und die besten **{r['k']}** dem Modell wörtlich in den Prompt gelegt. "
         "Das Modell antwortet *nur* darauf — es hat diese Texte nicht „gelernt“, "
         "es liest sie zur Laufzeit.", ""]
    for i, c in enumerate(r["chunks"], 1):
        L += [f"### Ausschnitt {i} — Tier {c['tier']} · Relevanz {c['score']}",
              f"**Quelle:** {c['title']}  ",
              f"`{c['source']}`", "",
              f"> {c['excerpt']}…", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--scope", default="bund")
    ap.add_argument("--out"); ap.add_argument("--json")
    a = ap.parse_args()
    r = run(a.question, a.scope)
    md = to_markdown(r)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md)
        print(f"[ok] Markdown → {a.out}", file=sys.stderr)
    if a.json:
        open(a.json, "w", encoding="utf-8").write(json.dumps(r, ensure_ascii=False, indent=2))
        print(f"[ok] JSON → {a.json}", file=sys.stderr)
    if not a.out:
        print(md)


if __name__ == "__main__":
    main()
