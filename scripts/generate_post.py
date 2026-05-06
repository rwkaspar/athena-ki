#!/usr/bin/env python3
"""
Athena - Social Media Stellungnahmen generieren

Verwendung:
    python generate_post.py "Die Bundesregierung plant eine Erhöhung der Mehrwertsteuer"
    python generate_post.py --topic "Rentenpolitik" --platform all
"""

import argparse
import json
import os
import sys
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

from critique import create_critique_chain
from retrieval import format_docs, get_vectorstores, tier_aware_retrieve

# Konfiguration
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "posts")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_MODEL = "athena"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RETRIEVER_K = 5
RETRIEVER_FETCH_K = 20

PLATFORM_PROMPTS = {
    "twitter": """Verfasse einen Tweet (maximal 280 Zeichen) zum Thema.

Inhalt:
- Eine prägnante, belegbare Faktaussage aus dem Kontext (eine Zahl, eine Norm, ein Fakt)
- Den zentralen Trade-off oder die offene Wertfrage benennen — KEINE Position, KEINE Empfehlung
- Statt "wir sollten" → "Spannungsfeld:", "Trade-off:", "offene Frage:"

Wenn der Kontext keine belastbare Faktaussage hergibt, schreib kurz und transparent, dass die Datenlage dünn ist.

Thema: {topic}

Antworte NUR mit dem Tweet-Text, ohne Anführungszeichen oder Erklärungen.""",

    "linkedin": """Verfasse eine LinkedIn-Analyse (300-600 Wörter) zum Thema im Stil einer strukturierten Optionsanalyse.

Struktur:
1. Faktenlage — mit konkreten Zahlen/Normen aus dem Kontext, mit Quellenangabe
2. Rechtsrahmen — relevante Vorschriften (BayGO, Satzungen, Bundes-/Landesrecht)
3. Handlungsoptionen — zwei bis vier Optionen mit ihren expliziten Trade-offs
4. Wertannahmen — was setzt jede Option implizit voraus?
5. Offene Fragen — was lässt sich aus dem Kontext NICHT beantworten?

Schließe NICHT mit einer Empfehlung oder einem "Fazit, das eine Option bevorzugt". Wenn der Kontext eine Stufe nicht stützt (z. B. keine Vergleichsfälle), benenne das transparent statt zu erfinden.

Thema: {topic}""",

    "instagram": """Verfasse einen Instagram-Post (150-300 Wörter) für alle Bürgerinnen und Bürger verständlich.

Inhalt:
- Konkrete Zahlen oder Fakten aus dem Kontext (mit kurzem Quellenhinweis)
- Den zentralen Konflikt oder die offene Wertfrage des Themas benennen — bewusst OHNE Position zu beziehen
- Kein "wir müssen" oder "die richtige Lösung ist", stattdessen: "die Frage ist", "abzuwägen ist"
- Ende mit relevanten Hashtags

Wenn der Kontext nichts Belastbares hergibt, sag das offen.

Thema: {topic}""",
}


RAG_PROMPT = PromptTemplate.from_template(
    """Nutze die folgenden Quellen aus der Wissensbasis als Grundlage. Wenn die
Quellen die Aufgabe nicht stützen, sag das offen statt zu erfinden.

Quellen:
{context}

Aufgabe:
{task}

Antwort:"""
)


def create_qa_chain(use_tier_boost: bool = True):
    """RAG-Pipeline als LCEL aufbauen — mit Hybrid-RAG (static + fresh) und
    Tier-aware Retrieval per Default."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    vectorstores = get_vectorstores(embeddings, CHROMA_DB_DIR)
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_HOST, timeout=600, reasoning=False)
    chain = RAG_PROMPT | llm | StrOutputParser()

    class _RagPipeline:
        def invoke(self, task: str):
            docs = tier_aware_retrieve(
                vectorstores, task,
                k=RETRIEVER_K, fetch_k=RETRIEVER_FETCH_K,
                use_tier_boost=use_tier_boost,
            )
            answer = chain.invoke({
                "context": format_docs(docs),
                "task": task,
            })
            return {"result": answer, "source_documents": docs}

    return _RagPipeline()


def generate_post(qa_chain, topic: str, platform: str) -> dict:
    """Stellungnahme für eine Plattform generieren. Rückgabe enthält Text
    und die retrieveten Quellen-Dokumente."""
    prompt = PLATFORM_PROMPTS[platform].format(topic=topic)
    result = qa_chain.invoke(prompt)
    return {"text": result["result"], "source_documents": result["source_documents"]}


def save_post(topic: str, posts: dict, critiques: dict | None = None, sources: list[str] | None = None):
    """Generierte Posts speichern, inkl. optional Critique pro Plattform und Quellen-URLs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{topic[:50].replace(' ', '_')}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    output = {
        "topic": topic,
        "generated_at": datetime.now().isoformat(),
        "posts": posts,
        "status": "draft",
    }
    if critiques:
        output["critiques"] = critiques
    if sources:
        output["sources"] = sources

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Gespeichert: {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="Athena - Social Media Stellungnahmen generieren"
    )
    parser.add_argument("topic", nargs="?", help="Das politische Thema")
    parser.add_argument(
        "--platform", "-p",
        choices=["twitter", "linkedin", "instagram", "all"],
        default="all",
        help="Zielplattform (default: all)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Posts nicht speichern, nur ausgeben",
    )
    parser.add_argument(
        "--no-tier-boost", action="store_true",
        help="Tier-Re-Ranking deaktivieren (reine Vektor-Ähnlichkeit) — für A/B-Vergleich",
    )
    parser.add_argument(
        "--critique", action="store_true",
        help="Pro Plattform-Post einen Critique-Pass laufen lassen (CRITIQUE_MODEL, default gemma3:27b). Verlängert Latenz pro Plattform.",
    )

    args = parser.parse_args()

    if not args.topic:
        parser.print_help()
        sys.exit(1)

    # Prüfe ob Datenbank existiert
    if not os.path.exists(CHROMA_DB_DIR):
        print("❌ Keine Wissensbasis gefunden!")
        print("   Bitte erst Dokumente einspeisen mit:")
        print("   python ingest.py --url https://www.gesetze-im-internet.de/gg/")
        sys.exit(1)

    flags = []
    flags.append("ohne Tier-Boost" if args.no_tier_boost else "mit Tier-Boost")
    if args.critique:
        flags.append("mit Critique-Pass")
    print(f"🔄 Lade Athena ({', '.join(flags)})...")
    qa_chain = create_qa_chain(use_tier_boost=not args.no_tier_boost)
    critique_fn = create_critique_chain() if args.critique else None

    platforms = (
        list(PLATFORM_PROMPTS.keys())
        if args.platform == "all"
        else [args.platform]
    )

    posts = {}
    critiques = {}
    sources_set = []

    for platform in platforms:
        print(f"\n📝 Generiere {platform.upper()}-Post...")
        print("⏳ Athena denkt nach...\n")
        gen = generate_post(qa_chain, args.topic, platform)
        posts[platform] = gen["text"]
        # Quellen über alle Plattformen aggregieren (Plattformen retrieven dasselbe k=5)
        for d in gen["source_documents"]:
            src = d.metadata.get("source", "?")
            if src not in sources_set:
                sources_set.append(src)

        print(f"{'=' * 60}")
        print(f"🏛️  ATHENA ({platform.upper()}):")
        print(f"{'=' * 60}")
        print(gen["text"])
        print(f"{'=' * 60}")

        if critique_fn is not None:
            print(f"⏳ Critique-Pass für {platform.upper()}...\n")
            critique = critique_fn(args.topic, gen["source_documents"], gen["text"])
            critiques[platform] = critique
            print(f"🔍 CRITIQUE ({platform.upper()}):")
            print(f"{'=' * 60}")
            print(critique)
            print(f"{'=' * 60}")

    print(f"\n📚 Genutzte Quellen ({len(sources_set)} unique):")
    for s in sources_set:
        print(f"   - {s}")

    if not args.no_save:
        save_post(args.topic, posts, critiques=critiques or None, sources=sources_set)


if __name__ == "__main__":
    main()
