#!/usr/bin/env python3
"""Vollständiger Athena-Pipeline-Durchlauf (BUND/EVIDENZ) als Transparenz-Beispiel.

Zeigt ALLE Stufen an EINER Frage — auf dem Bundes-Korpus (nicht dem Pfofeld-Pilot
von query.py):
  1. Hybrid-RAG-Retrieval (Tier-Re-Ranking)        → welche Quellen-Ausschnitte
  2. Antwort (Athena, faktenbasiert, keine Empfehlung)
  3. Strukturierte Optionsanalyse (Pydantic)        → Faktenlage/Optionen/Trade-offs
  4. Faktentreue-Verifikation (verify_claims)        → jede Aussage gegen Tier-1
  5. Critique-Pass (Devil's Advocate, anderes Modell) → adversariale Gegenprüfung

Aufruf:
    OLLAMA_HOST=… MISTRAL_API_KEY=… STRUCTURE_MODEL=gemma4:26b CRITIQUE_MODEL=gemma4:26b \
    VERIFY_MODEL=gemma4:26b python scripts/pipeline_demo.py "Frage" --out f.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ANSWER_PROMPT = """Du bist Athena, die faktenbasierte Analyse-KI von EVIDENZ. Beantworte die Frage \
AUSSCHLIESSLICH auf Basis der folgenden Quellen — keine Empfehlung, keine erfundenen Zahlen. \
Struktur: (1) Faktenlage mit Quellenbezug, (2) Rechtsrahmen, (3) Handlungsoptionen mit \
ehrlich benannten Trade-offs, (4) offene Fragen/Datenlücken.

QUELLEN:
{context}

FRAGE: {question}
"""


def _ser(o):
    if o is None:
        return None
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "dict"):
        return o.dict()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--scope", default="bund")
    ap.add_argument("--sim-floor", type=float, default=0.45)
    ap.add_argument("--max-k", type=int, default=15)  # Positions-Pipeline: breit
    ap.add_argument("--out")
    a = ap.parse_args()

    import serve
    from retrieval import tier_aware_retrieve, format_docs
    from structure import structure_analysis
    from verify import verify_claims
    from critique import create_critique_chain
    from langchain_ollama import OllamaLLM

    print("… Stufe 1: Retrieval (Bund)", file=sys.stderr)
    vs, _ = serve._get_components(a.scope)
    docs = tier_aware_retrieve(vs, a.question, k=serve.RETRIEVER_K, fetch_k=50,
                               sim_floor=a.sim_floor, max_k=a.max_k)

    print("… Stufe 2: Antwort (Ollama, lokal)", file=sys.stderr)
    llm = OllamaLLM(model=os.getenv("ANSWER_MODEL", "gemma4:31b"),
                    base_url=os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434"),
                    timeout=600, reasoning=False)
    answer = llm.invoke(ANSWER_PROMPT.format(context=format_docs(docs), question=a.question))

    print("… Stufe 3: Strukturierte Optionsanalyse", file=sys.stderr)
    analysis = structure_analysis(answer)
    print("… Stufe 4: Faktentreue-Verifikation", file=sys.stderr)
    analysis = verify_claims(analysis, docs)
    print("… Stufe 5: Critique-Pass (Devil's Advocate)", file=sys.stderr)
    critique = create_critique_chain()(a.question, docs, answer)

    out = {
        "question": a.question, "scope": a.scope,
        "k": serve.RETRIEVER_K, "fetch_k": serve.RETRIEVER_FETCH_K,
        "retrieval": [{
            "tier": d.metadata.get("tier_rank"),
            "score": d.metadata.get("_combined_score"),
            "title": d.metadata.get("title") or d.metadata.get("source"),
            "source": d.metadata.get("source"),
            "excerpt": " ".join((d.page_content or "").split())[:240],
        } for d in docs],
        "answer": answer,
        "analysis": _ser(analysis),
        "critique": critique,
    }
    js = json.dumps(out, ensure_ascii=False, indent=2, default=str)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(js)
        print(f"[ok] → {a.out}", file=sys.stderr)
    else:
        print(js)


if __name__ == "__main__":
    main()
