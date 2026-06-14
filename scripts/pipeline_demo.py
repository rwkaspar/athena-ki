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


def adversarial_verify(facts, context, host):
    """Adversarial Verification: ein UNABHÄNGIGES Modell (andere Familie als die
    Antwort) versucht AKTIV, jede Aussage zu widerlegen — nur anhand der Quellen.
    Hält die Aussage dem Widerlegungsversuch stand? Im Zweifel skeptisch. Das ist
    schärfer als die Faktentreue-Prüfung (stützt die Quelle?) und der holistische
    Critique-Pass — es ist der gezielte Angriff auf jede Einzelaussage."""
    import json as _j
    import re as _re
    from langchain_ollama import OllamaLLM
    m = OllamaLLM(model=os.getenv("ADVERSARIAL_MODEL", "mistral-small3.2:latest"),
                  base_url=host, timeout=300, reasoning=False)
    out = []
    for f in facts:
        aussage = f.get("aussage", "")
        prompt = (
            "Du bist ein extrem skeptischer Gegen-Prüfer. Versuche AKTIV, die folgende "
            "Aussage zu WIDERLEGEN — ausschließlich anhand der Quellen. Stützen die Quellen "
            "sie klar und du kannst sie nicht widerlegen: verdict \"haelt\". Widersprechen die "
            "Quellen ihr oder belegen sie nicht: \"widerlegt\". Unklar: \"zweifelhaft\". "
            "Im Zweifel skeptisch.\n"
            "Antworte NUR als JSON: {\"verdict\":\"haelt|widerlegt|zweifelhaft\",\"begruendung\":\"<1 Satz>\"}\n\n"
            f"QUELLEN:\n{context}\n\nAUSSAGE: {aussage}"
        )
        try:
            raw = m.invoke(prompt)
            js = _re.search(r"\{.*\}", raw, _re.S)
            d = _j.loads(js.group(0)) if js else {}
        except Exception:
            d = {"verdict": "zweifelhaft", "begruendung": "Gegenprüfung nicht abgeschlossen."}
        v = (d.get("verdict") or "zweifelhaft").strip().lower()
        if v not in ("haelt", "widerlegt", "zweifelhaft"):
            v = "zweifelhaft"
        out.append({"aussage": aussage, "verdict": v, "begruendung": d.get("begruendung", "")})
    return out


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
    from langchain_mistralai import ChatMistralAI

    print("… Stufe 1: Retrieval (Bund)", file=sys.stderr)
    vs, _ = serve._get_components(a.scope)
    docs = tier_aware_retrieve(vs, a.question, k=serve.RETRIEVER_K, fetch_k=50,
                               sim_floor=a.sim_floor, max_k=a.max_k)

    # Position/Antwort = Mistral (das echte Positions-Modell), läuft über die API
    # (nicht auf der iGPU). Critique = athena (qwen-basiert), Adversarial = gemma4.
    print("… Stufe 2: Position (Mistral)", file=sys.stderr)
    llm = ChatMistralAI(model="mistral-large-latest", api_key=os.environ["MISTRAL_API_KEY"],
                        temperature=0.2, max_tokens=2200, timeout=180)
    answer = llm.invoke(ANSWER_PROMPT.format(context=format_docs(docs), question=a.question)).content

    print("… Stufe 3: Strukturierte Optionsanalyse", file=sys.stderr)
    analysis = structure_analysis(answer)
    print("… Stufe 4: Faktentreue-Verifikation", file=sys.stderr)
    analysis = verify_claims(analysis, docs)
    analysis_d = _ser(analysis) or {}
    print("… Stufe 5: Adversarial Verification (unabhängiges Modell)", file=sys.stderr)
    host = os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434")
    adversarial = adversarial_verify(analysis_d.get("faktenlage", []), format_docs(docs), host)
    print("… Stufe 6: Critique-Pass (Devil's Advocate)", file=sys.stderr)
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
        "analysis": analysis_d,
        "adversarial": adversarial,
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
