"""Athena — Quellen-Einfluss via Leave-one-out-Ablation (Stufe 3).

Misst den *kausalen* Einfluss einer Quelle auf eine Optionsanalyse: Wie stark
ändert sich die EVIDENZ-Position, wenn genau diese Quelle aus dem Kontext
entfernt wird? Das ist der einzige ehrliche „Einfluss"-Begriff — Retrieval-Score
misst nur Relevanz (passte zur Frage), nicht Wirkung (prägte die Entscheidung).

SKALIERUNG — der entscheidende Punkt:
  Ablation läuft NICHT über den ganzen Korpus, sondern nur über die Quellen, die
  für DIESE Frage tatsächlich abgerufen wurden (top-k). Eine nie abgerufene
  Quelle war nie im Prompt → Einfluss per Definition null. Damit skaliert der
  Aufwand mit k (≈5–20), nicht mit der Korpusgröße (10⁴+).

KOSTEN:
  1 Baseline-Lauf + N Ablations-Läufe, N = Zahl der getesteten Quellen.
  Mit --top-n wird N gedeckelt: nur die einflussreichsten Kandidaten (höchster
  Re-Rank-Score) werden ablatiert, der Rest wird als „nicht getestet" EHRLICH
  protokolliert — kein stilles Weglassen.

  Offline-Batch-Job, kein Live-Request. Braucht aitest (OLLAMA_HOST) online,
  außer im --dry-run (zeigt Plan + Kostenschätzung ohne LLM-Aufrufe).

AUSGABE:
  JSON-Report pro Frage: baseline-Position, pro Quelle der position_delta in
  [0,1] (0 = kein Einfluss, 1 = Position komplett anders), plus Metadaten.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from retrieval import format_docs, get_vectorstores, tier_aware_retrieve

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RETRIEVER_K = int(os.getenv("ABLATION_K", "8"))
RETRIEVER_FETCH_K = 20

LLM_MODEL_FOR_SCOPE = {
    "pfofeld": "athena-pfofeld",
    "bund": "athena",
}

PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Quellen aus der kuratierten Wissensbasis (Tier-klassifiziert, versioniert):

{context}

Frage: {question}"""
)

# Wir vergleichen die ENTSCHEIDUNG, nicht den ganzen Text. Die Position ist das,
# was sich ändern können soll — die Faktenlage bleibt ja oft gleich.
POSITION_HEADINGS = ("EVIDENZ-Position", "EVIDENZ-Position", "Position", "Empfehlung")


def extract_position(text: str) -> str:
    """Zieht den EVIDENZ-Positions-Abschnitt aus einer Analyse. Fällt auf den
    Gesamttext zurück, wenn keine Überschrift gefunden wird."""
    lines = text.splitlines()
    capture = False
    out = []
    for line in lines:
        if re.match(r"^#+\s", line):
            heading = re.sub(r"^#+\s*", "", line).strip().lower()
            if any(h.lower() in heading for h in POSITION_HEADINGS):
                capture = True
                continue
            if capture:
                # nächste gleich-/höherrangige Überschrift beendet den Abschnitt,
                # außer es ist eine Unterüberschrift (Begründung/Mapping gehören dazu)
                if not heading.startswith(("begründ", "mapping", "trade", "konfidenz")):
                    break
        elif capture:
            out.append(line)
    section = "\n".join(out).strip()
    return section or text.strip()


def cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def source_label(doc):
    m = doc.metadata
    return {
        "source": m.get("source", "?"),
        "title": m.get("title"),
        "tier_rank": m.get("tier_rank"),
        "tier_label": m.get("tier_label"),
        "collection": m.get("_collection"),
        "similarity": m.get("_similarity"),
        "combined_score": m.get("_combined_score"),
    }


def run_ablation(question, scope, top_n, dry_run):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    vectorstores = get_vectorstores(embeddings, CHROMA_DB_DIR, scope=scope)

    # 1. Retrieval EINMAL — das sind die einzigen Quellen mit möglichem Einfluss.
    docs = tier_aware_retrieve(
        vectorstores, question, k=RETRIEVER_K, fetch_k=RETRIEVER_FETCH_K,
        use_tier_boost=True,
    )
    # Kandidaten zum Ablatieren: top-n nach Re-Rank-Score, Rest ehrlich vermerken.
    tested = docs[:top_n] if top_n else docs
    skipped = docs[top_n:] if top_n else []

    plan = {
        "question": question,
        "scope": scope,
        "retrieved": len(docs),
        "to_ablate": len(tested),
        "skipped_low_score": [source_label(d) for d in skipped],
        "llm_runs_needed": 1 + len(tested),
    }

    if dry_run:
        plan["note"] = (
            "DRY RUN — keine LLM-Aufrufe. Würde 1 Baseline + "
            f"{len(tested)} Ablations-Läufe brauchen. Skaliert mit k, nicht Korpus."
        )
        return plan

    llm = OllamaLLM(model=LLM_MODEL_FOR_SCOPE[scope], base_url=OLLAMA_HOST,
                    timeout=600, reasoning=False)
    chain = PROMPT_TEMPLATE | llm | StrOutputParser()

    # 2. Baseline mit allen abgerufenen Quellen.
    t0 = time.time()
    baseline_text = chain.invoke({"context": format_docs(docs), "question": question})
    baseline_pos = extract_position(baseline_text)
    baseline_emb = embeddings.embed_query(baseline_pos)

    # 3. Pro getesteter Quelle: Re-Run OHNE diese Quelle, Positions-Delta messen.
    influences = []
    for i, doc in enumerate(tested):
        subset = [d for j, d in enumerate(docs) if d is not doc]
        ablated_text = chain.invoke({"context": format_docs(subset), "question": question})
        ablated_pos = extract_position(ablated_text)
        ablated_emb = embeddings.embed_query(ablated_pos)
        delta = round(1.0 - cosine(baseline_emb, ablated_emb), 4)
        info = source_label(doc)
        info["position_delta"] = delta  # 0 = kein Einfluss … 1 = Position völlig anders
        influences.append(info)
        print(f"  [{i+1}/{len(tested)}] Δ={delta}  {info['source']}", file=sys.stderr)

    influences.sort(key=lambda x: -x["position_delta"])
    return {
        **plan,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": LLM_MODEL_FOR_SCOPE[scope],
        "elapsed_s": round(time.time() - t0, 1),
        "baseline_position": baseline_pos[:2000],
        "influences": influences,
        "method": "leave-one-out ablation; delta = 1 - cosine(embed(baseline_position), embed(ablated_position))",
    }


def main():
    p = argparse.ArgumentParser(description="Quellen-Einfluss via Leave-one-out-Ablation")
    p.add_argument("question", help="Die Entscheidungsfrage der Analyse")
    p.add_argument("--scope", default="bund", choices=list(LLM_MODEL_FOR_SCOPE))
    p.add_argument("--top-n", type=int, default=8,
                   help="Nur die N einflussreichsten abgerufenen Quellen ablatieren (Kostenbremse). 0 = alle.")
    p.add_argument("--dry-run", action="store_true",
                   help="Nur Plan + Kostenschätzung, keine LLM-Aufrufe (offline testbar).")
    p.add_argument("--out", help="JSON-Report in diese Datei schreiben")
    args = p.parse_args()

    report = run_ablation(args.question, args.scope, args.top_n or None, args.dry_run)
    out_json = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"[ok] Report → {args.out}", file=sys.stderr)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
