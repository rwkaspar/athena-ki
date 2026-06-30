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
strukturiert: (1) Faktenlage, (2) Rechtsrahmen, (3) Handlungsoptionen mit ehrlich benannten \
Trade-offs, (4) offene Fragen. Keine Empfehlung.

STRIKTE QUELLENBINDUNG: Nutze AUSSCHLIESSLICH Aussagen, die direkt aus den unten gelieferten \
Quellen belegbar sind, und zitiere dahinter den Paragraphen/die Quelle wörtlich (z. B. „(Art. 16 \
KI-VO)"). Nenne KEINE Zahl, kein Datum, keinen Paragraphen und keinen Eigennamen, der nicht in den \
Quellen steht — ergänze NICHTS aus deinem Vorwissen. Was die Quellen nicht hergeben, kommt unter \
(4) als offene Frage — NICHT erfinden, NICHT aus dem Gedächtnis ergänzen. Lieber wenige, voll \
belegte Aussagen als viele unbelegte.

QUELLEN:
{context}
{canon_block}
FRAGE: {question}
"""

CANON_BLOCK = """
WERTEKANON-BEWERTUNG: Beurteile für JEDE Handlungsoption zusätzlich, wie sie auf die \
sieben Paragraphen des EVIDENZ-Wertekanons wirkt. Gib das pro Option als kleine Tabelle \
mit Zeilen „§N: <Wert von -100 bis +100> — <kurze Begründung>" aus. +100 = die Option \
stützt diesen Wert stark, 0 = neutral, -100 = sie belastet ihn stark. Das ist KEINE \
Empfehlung — es macht nur transparent, welche Werte eine Option berührt. Bewerte alle \
§1–§7; was eine Option nicht berührt, ist 0.

WERTEKANON (§§1–7):
{canon}
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
    m = OllamaLLM(model=os.getenv("ADVERSARIAL_MODEL", "gemma4:26b"),
                  base_url=host, timeout=300, reasoning=False)
    out = []
    for f in facts:
        aussage = f.get("aussage", "")
        prompt = (
            "Du bist ein Gegen-Prüfer (Adversarial Verification). Deine EINZIGE Aufgabe: "
            "Kannst du die folgende Aussage anhand der Quellen WIDERLEGEN? "
            "Verdikte: \"widerlegt\" = die Quellen widersprechen der Aussage klar (echte "
            "Gegen-Evidenz). \"zweifelhaft\" = die Quellen stellen sie teilweise in Frage. "
            "\"haelt\" = die Quellen widersprechen ihr NICHT (sie stützen sie oder sagen nichts "
            "Gegenteiliges). WICHTIG: Nicht-Finden ist KEINE Widerlegung — wenn du keine "
            "Gegen-Evidenz in den Quellen findest, lautet das Verdikt \"haelt\". Nur echter "
            "Widerspruch flaggt.\n"
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
        out.append({"aussage": aussage, "verdict": v, "begruendung": d.get("begruendung", ""),
                    "quelle": (d.get("quelle") or "").strip()})
    return out


def critique_verdict(critique_text, host):
    """Fasst den (langen) Critique-Pass in EINE faire Verdikt-Zeile zusammen — für die
    Methodik-Seite. Anderes Modell (gemma4) als der Critique selbst. Unterscheidet
    inhaltlich falsche Aussagen (= Faktenfehler) von bloßen Präzisions-/Methodenhinweisen,
    damit ein gründlicher Review nicht als „durchgefallen" missverstanden wird."""
    import json as _j
    import re as _re
    from langchain_ollama import OllamaLLM
    if not (critique_text or "").strip():
        return {}
    m = OllamaLLM(model=os.getenv("VERDICT_MODEL", "gemma4:26b"),
                  base_url=host, timeout=300, num_ctx=16384, num_predict=400, reasoning=False)
    prompt = (
        "Du fasst das Ergebnis eines methodischen Critique-Reviews einer Optionsanalyse in EIN "
        "faires Verdikt zusammen. Die ENTSCHEIDENDE Frage für eine faktenbasierte Politik-KI ist: "
        "Wurden Fakten/Zahlen ERFUNDEN — also Aussagen behauptet, die in KEINER der Quellen stehen "
        "(Halluzination)?\n"
        "WICHTIGE Unterscheidung: Eine falsche ZUORDNUNG von Inhalten, die sehr wohl in den Quellen "
        "stehen (z. B. der richtige Artikel/Absatz, die richtige Rolle), ist KEINE Erfindung, "
        "sondern ein Präzisions-/Zuordnungshinweis. Ebenso sind Methoden- oder Formulierungshinweise "
        "KEINE Erfindung. Ein gründlicher Review findet fast immer solche Hinweise — das ist gewollt "
        "und kein Durchfallen. Nur eine echte Erfindung (unbelegte Behauptung) ist ein Faktenfehler.\n"
        "Antworte NUR als JSON: {\"erfundene_fakten\": true|false (wurden unbelegte, in KEINER Quelle "
        "stehende Aussagen/Zahlen erfunden?), \"icon\": \"\\u2705\" wenn KEINE erfundenen Fakten "
        "(auch wenn es Präzisionshinweise gibt), sonst \"\\u26a0\\ufe0f\", \"fazit\": \"<EIN "
        "stützender, ehrlicher Satz: zuerst, dass keine Fakten erfunden wurden und alle Aussagen "
        "quellenbasiert sind (falls zutreffend), dann knapp, welche Präzisierungen der Review "
        "anmerkt>\"}\n\n"
        f"CRITIQUE:\n{critique_text}"
    )
    try:
        raw = m.invoke(prompt)
        js = _re.search(r"\{.*\}", raw, _re.S)
        d = _j.loads(js.group(0)) if js else {}
    except Exception:
        return {}
    if not d.get("fazit"):
        return {}
    d["icon"] = d.get("icon") or ("⚠️" if d.get("erfundene_fakten") else "✅")
    return d


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
    ap.add_argument("--value-canon", action="store_true",
                    help="Wertekanon §§1–7 beigeben → pro Option Wert-Wirkung (-100..100)")
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
                        temperature=0.0, max_tokens=2800, timeout=180)
    canon_block = ""
    if a.value_canon:
        canon_path = os.path.join(os.path.dirname(__file__), "..", "config", "wertekanon.md")
        try:
            canon_txt = open(canon_path, encoding="utf-8").read()
            canon_block = CANON_BLOCK.format(canon=canon_txt)
        except OSError as e:
            print(f"[warn] Wertekanon nicht ladbar: {e}", file=sys.stderr)
    answer = llm.invoke(ANSWER_PROMPT.format(context=format_docs(docs), question=a.question,
                                             canon_block=canon_block)).content

    print("… Stufe 3: Strukturierte Optionsanalyse", file=sys.stderr)
    analysis = structure_analysis(answer)
    print("… Stufe 4: Faktentreue-Verifikation", file=sys.stderr)
    analysis = verify_claims(analysis, docs)
    analysis_d = _ser(analysis) or {}
    print("… Stufe 5: Adversarial Verification (unabhängiges Modell)", file=sys.stderr)
    host = os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434")
    adversarial = adversarial_verify(analysis_d.get("faktenlage", []), format_docs(docs), host)
    print("… Stufe 6: Critique-Pass (Devil's Advocate)", file=sys.stderr)
    # Scope-richtiges Critique-Modell: athena (föderales Standardmodell) für bund, sonst
    # athena-pfofeld (Pilot mit eingebranntem Gemeinde-Pfofeld-System-Prompt, der eine
    # Bundes-Analyse fälschlich an „Gemeinde Pfofeld" messen würde).
    crit_model = os.getenv("CRITIQUE_MODEL") or (
        "athena:latest" if a.scope == "bund" else "athena-pfofeld:latest")
    critique = create_critique_chain(model=crit_model)(a.question, docs, answer)
    print("… Stufe 7: Critique-Verdikt (faire Zusammenfassung)", file=sys.stderr)
    verdict = critique_verdict(critique, host)

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
        "critique_verdict": verdict,
    }
    js = json.dumps(out, ensure_ascii=False, indent=2, default=str)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(js)
        print(f"[ok] → {a.out}", file=sys.stderr)
    else:
        print(js)


if __name__ == "__main__":
    main()
