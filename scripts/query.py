#!/usr/bin/env python3
"""
Athena - Wissensbasis abfragen

Verwendung:
    python query.py "Was sagt das Grundgesetz zur Menschenwürde?"
    python query.py --interactive
"""

import argparse
import time
import os
import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from critique import create_critique_chain
from notion_sink import publish_to_notion
from retrieval import format_docs, get_vectorstores, tier_aware_retrieve
from structure import structure_analysis
from verify import verify_claims

# Konfiguration
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_MODEL = "athena"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RETRIEVER_K = 5
RETRIEVER_FETCH_K = 20  # Kandidaten-Pool vor dem Tier-Re-Ranking

PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Nutze die folgenden Quellen aus der Wissensbasis, um die Frage zu beantworten.
Wenn die Quellen keine belastbare Antwort enthalten, sag das offen und erfinde nichts.

Quellen:
{context}

Frage: {question}
Antwort:"""
)


def create_qa_chain(
    use_tier_boost: bool = True,
    use_critique: bool = False,
    use_structure: bool = False,
    use_verify: bool = False,
):
    """Stage-4-Pipeline: Hybrid-RAG → strukturierte Optionsanalyse (JSON nach
    pydantic-Schema) → optional Critique-Pass. Rückgabe ist Objekt mit
    .invoke(question) → {"analysis": Optionsanalyse, "rendered": str,
    "source_documents": list[, "critique": str]}."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    vectorstores = get_vectorstores(embeddings, CHROMA_DB_DIR)
    # reasoning=False weil qwen3.6's Thinking-Mode mit langchain-ollama-Streaming
    # nicht zuverlässig response füllt (siehe claude.md). Output bleibt
    # strukturiertes Markdown nach den 6 Stufen — das Modelfile-SYSTEM erzwingt
    # die Struktur. Strikte JSON-Erzwingung über Ollamas format-Param ist in
    # 0.18.3 unzuverlässig (Schema wird nicht konsistent angewendet); JSON kommt
    # später via Post-Processing (siehe scripts/schema.py als Zielform).
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_HOST, timeout=600, reasoning=False)
    answer_chain = PROMPT_TEMPLATE | llm | StrOutputParser()
    critique_fn = create_critique_chain() if use_critique else None

    class _RagPipeline:
        def invoke(self, question: str):
            docs = tier_aware_retrieve(
                vectorstores, question,
                k=RETRIEVER_K, fetch_k=RETRIEVER_FETCH_K,
                use_tier_boost=use_tier_boost,
            )
            answer = answer_chain.invoke({
                "context": format_docs(docs),
                "question": question,
            })
            result = {"result": answer, "source_documents": docs}
            if use_structure:
                result["analysis"] = structure_analysis(answer)
                if use_verify:
                    result["analysis"] = verify_claims(result["analysis"], docs)
            if critique_fn is not None:
                result["critique"] = critique_fn(question, docs, answer)
            return result

    return _RagPipeline()


def ask(qa_chain, question: str):
    """Eine Frage stellen und Antwort mit Quellen (und optional Critique) ausgeben."""
    print(f"\n🔍 Frage: {question}")
    print("⏳ Athena denkt nach...\n")
    start_time = time.time()

    result = qa_chain.invoke(question)

    print("=" * 60)
    print("🏛️  ATHENA:")
    print("=" * 60)
    print(result["result"])

    if "analysis" in result:
        print("\n" + "=" * 60)
        print("📐 STRUCTURED (Pydantic-validiertes JSON):")
        print("=" * 60)
        print(result["analysis"].model_dump_json(indent=2))

    if "critique" in result:
        print("\n" + "=" * 60)
        print("🔍 CRITIQUE-PASS (Devil's Advocate, anderes Modell):")
        print("=" * 60)
        print(result["critique"])

    print("\n" + "-" * 60)
    elapsed = time.time() - start_time
    print(f"\n⏱️  Antwortzeit: {elapsed:.1f} Sekunden")
    print("\n" + "-" * 60)
    print("📚 Quellen (sortiert nach Re-Rank-Score):")
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "Unbekannt")
        rank = doc.metadata.get("tier_rank", "?")
        label = doc.metadata.get("tier_label", "?")
        collection = doc.metadata.get("_collection", "?")
        sim = doc.metadata.get("_similarity")
        combined = doc.metadata.get("_combined_score")
        score_str = (
            f"  sim={sim} → rank={combined}"
            if sim is not None and combined is not None
            else ""
        )
        print(f"   [{i}] [{collection}] Tier {rank} ({label}){score_str}")
        print(f"       {source}")
    print("=" * 60)

    return result


def interactive_mode(qa_chain):
    """Interaktiver Modus für mehrere Fragen."""
    print("\n🏛️  Athena - Interaktiver Modus")
    print("   Stelle Fragen oder tippe 'exit' zum Beenden.\n")

    while True:
        try:
            question = input("\n❓ Deine Frage: ").strip()
            if question.lower() in ("exit", "quit", "q", "bye"):
                print("\n👋 Auf Wiedersehen!")
                break
            if not question:
                continue
            ask(qa_chain, question)
        except KeyboardInterrupt:
            print("\n\n👋 Auf Wiedersehen!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Athena - Wissensbasis abfragen"
    )
    parser.add_argument("question", nargs="?", help="Die Frage an Athena")
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Interaktiver Modus für mehrere Fragen"
    )
    parser.add_argument(
        "--no-tier-boost", action="store_true",
        help="Tier-Re-Ranking deaktivieren (reine Vektor-Ähnlichkeit) — für A/B-Vergleich",
    )
    parser.add_argument(
        "--critique", action="store_true",
        help="Critique-Pass aktivieren — zweites LLM (CRITIQUE_MODEL, default gemma3:27b) prüft die Antwort. Verdoppelt Latenz.",
    )
    parser.add_argument(
        "--structure", action="store_true",
        help="Stage 4: Markdown-Antwort zusätzlich in Pydantic-validiertes JSON konvertieren (STRUCTURE_MODEL, default gemma3:27b). Notwendig für Notion-Sink und strukturierten Downstream-Konsum.",
    )
    parser.add_argument(
        "--notion", action="store_true",
        help="Stage 6: Optionsanalyse als Notion-Page in den 🧭 Optionsanalysen-Subtree pushen. Impliziert --structure.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Stage 3: Jede Aussage in der Faktenlage einzeln gegen Tier-1-Quellen prüfen (VERIFY_MODEL, default gemma3:27b). Setzt verification_status und evidence_quote pro Faktum. Impliziert --structure.",
    )

    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.print_help()
        sys.exit(1)

    # Prüfe ob Datenbank existiert
    if not os.path.exists(CHROMA_DB_DIR):
        print("❌ Keine Wissensbasis gefunden!")
        print("   Bitte erst Dokumente einspeisen mit:")
        print("   python ingest.py --url https://www.gesetze-im-internet.de/gg/")
        sys.exit(1)

    # --notion und --verify impliziern beide --structure
    use_structure = args.structure or args.notion or args.verify
    flags = []
    flags.append("ohne Tier-Boost" if args.no_tier_boost else "mit Tier-Boost")
    if use_structure:
        flags.append("mit Structure")
    if args.verify:
        flags.append("mit Verify")
    if args.critique:
        flags.append("mit Critique-Pass")
    if args.notion:
        flags.append("mit Notion-Sink")
    print(f"🔄 Lade Athena ({', '.join(flags)})...")
    qa_chain = create_qa_chain(
        use_tier_boost=not args.no_tier_boost,
        use_critique=args.critique,
        use_structure=use_structure,
        use_verify=args.verify,
    )

    if args.interactive:
        interactive_mode(qa_chain)
    else:
        result = ask(qa_chain, args.question)
        if args.notion:
            print("\n" + "=" * 60)
            print("📤 PUSHE NACH NOTION...")
            try:
                sources = [d.metadata.get("source", "?") for d in result["source_documents"]]
                url = publish_to_notion(
                    question=args.question,
                    analysis=result["analysis"],
                    sources=sources,
                    critique=result.get("critique"),
                )
                print(f"   ✓ Page erstellt: {url}")
            except Exception as e:
                print(f"   ❌ Push fehlgeschlagen: {e}")
            print("=" * 60)


if __name__ == "__main__":
    main()
