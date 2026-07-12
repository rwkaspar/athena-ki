#!/usr/bin/env python3
"""Faktencheck-Pass für die Steuersystem-Analyse (Stufe a).

Unabhängige Familie (gemma4, nicht der Mistral-Erzeuger) prüft jede empirische
Behauptung der Analyse gegen die RAG-Quellen und listet die NICHT belegten
(Zahlen, %, €, Statistiken, benannte Quellen/Zitate). Markiert sie inline als
⟦unbelegt: …⟧ und schreibt eine Prüfliste.
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_steuersystem import QUERIES

PROMPT = """Du bist ein strenger Faktenprüfer. Unten stehen QUELLEN und eine ANALYSE. Liste JEDE empirische \
Tatsachenbehauptung der ANALYSE, die NICHT durch die QUELLEN gedeckt ist — konkrete Zahlen, Prozentwerte, \
€-Beträge, Aufkommenszahlen, Steuersätze, Statistiken, benannte Quellen/Institutionen mit Jahr (z. B. \
„Bundesrechnungshof 2021") und wörtliche Zitate. Gib den betreffenden Textteil WÖRTLICH und exakt so zurück, \
wie er in der ANALYSE steht (kopierfähige Teil-Zeichenkette). NICHT listen: qualitative Bewertungen, \
Modellbeschreibungen, Wertannahmen — nur unbelegte empirische FAKTEN/Zahlen/Quellenangaben.

Antworte NUR als JSON: {"fabrikate":[{"zitat":"<wörtlich aus der ANALYSE>","art":"zahl|quelle|zitat","grund":"<kurz>"}]}

QUELLEN:
{context}

ANALYSE:
{analyse}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="eval/steuersystem_analyse.md")
    ap.add_argument("--out", default="eval/steuersystem_analyse_markiert.md")
    ap.add_argument("--list", default="eval/steuersystem_pruefliste.json")
    a = ap.parse_args()

    import serve
    from retrieval import tier_aware_retrieve, format_docs
    from langchain_ollama import OllamaLLM
    vs, _ = serve._get_components("bund")
    docs, seen = [], set()
    for q in QUERIES:
        for d in tier_aware_retrieve(vs, q, k=serve.RETRIEVER_K, fetch_k=50, sim_floor=0.35, max_k=8):
            key = (d.metadata.get("source", ""), d.page_content[:80])
            if key not in seen:
                seen.add(key); docs.append(d)
    ctx = format_docs(docs)
    analyse = open(a.inp, encoding="utf-8").read()

    host = os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434")
    chk = OllamaLLM(model=os.getenv("ADVERSARIAL_MODEL", "gemma4:26b"), base_url=host,
                    timeout=1200, num_ctx=32768, num_predict=3000, reasoning=False, format="json")
    raw = chk.invoke(PROMPT.replace("{context}", ctx[:13000]).replace("{analyse}", analyse[:14000]))
    try:
        fab = json.loads(re.search(r"\{.*\}", raw, re.S).group(0)).get("fabrikate", [])
    except Exception as e:
        print(f"[fail] parse: {e}\n{raw[:500]}", file=sys.stderr); sys.exit(1)
    # nur behalten, was wörtlich im Text steht
    fab = [f for f in fab if f.get("zitat", "").strip() and f["zitat"] in analyse]
    marked = analyse
    for f in fab:
        z = f["zitat"]
        if z in marked and "⟦unbelegt:" not in marked[max(0, marked.find(z) - 12):marked.find(z)]:
            marked = marked.replace(z, f"⟦unbelegt: {z}⟧", 1)
    open(a.out, "w", encoding="utf-8").write(marked)
    json.dump(fab, open(a.list, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[ok] {len(fab)} unbelegte Behauptungen markiert → {a.out}", file=sys.stderr)
    for f in fab:
        print(f"  [{f.get('art','?')}] {f['zitat'][:75]}", file=sys.stderr)


if __name__ == "__main__":
    main()
