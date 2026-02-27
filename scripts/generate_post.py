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

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

# Konfiguration
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "posts")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "athena"

PLATFORM_PROMPTS = {
    "twitter": """Verfasse eine klare, faktenbasierte Stellungnahme zum folgenden Thema 
für X/Twitter (max. 280 Zeichen). Sei direkt und prägnant. 
Nutze wenn möglich eine konkrete Zahl oder Fakt.

Thema: {topic}

Antworte NUR mit dem Tweet-Text, ohne Anführungszeichen oder Erklärungen.""",

    "linkedin": """Verfasse eine professionelle, faktenbasierte Analyse zum folgenden Thema 
für LinkedIn (300-600 Wörter). Strukturiere den Text mit:
1. Kernaussage (1-2 Sätze)
2. Faktenbasierte Analyse (mit konkreten Zahlen/Daten)
3. Bewertung und Einordnung
4. Fazit

Thema: {topic}

Nutze Fakten und Daten aus dem bereitgestellten Kontext.""",

    "instagram": """Verfasse eine verständliche, faktenbasierte Stellungnahme zum folgenden Thema 
für Instagram (150-300 Wörter). Der Text sollte:
- Für alle Bürger verständlich sein
- Konkrete Zahlen nennen
- Eine klare Position beziehen
- Mit relevanten Hashtag-Vorschlägen enden

Thema: {topic}

Nutze Fakten und Daten aus dem bereitgestellten Kontext.""",
}


def create_qa_chain():
    """RAG-Chain erstellen."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )
    llm = OllamaLLM(model=LLM_MODEL, timeout=600)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    return qa_chain


def generate_post(qa_chain, topic: str, platform: str) -> str:
    """Stellungnahme für eine Plattform generieren."""
    prompt = PLATFORM_PROMPTS[platform].format(topic=topic)
    result = qa_chain.invoke(prompt)
    return result["result"]


def save_post(topic: str, posts: dict):
    """Generierte Posts speichern."""
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

    print("🔄 Lade Athena...")
    qa_chain = create_qa_chain()

    platforms = (
        list(PLATFORM_PROMPTS.keys())
        if args.platform == "all"
        else [args.platform]
    )

    posts = {}
    for platform in platforms:
        print(f"\n📝 Generiere {platform.upper()}-Post...")
        print("⏳ Athena denkt nach...\n")
        post = generate_post(qa_chain, args.topic, platform)
        posts[platform] = post

        print(f"{'=' * 60}")
        print(f"🏛️  ATHENA ({platform.upper()}):")
        print(f"{'=' * 60}")
        print(post)
        print(f"{'=' * 60}")

    if not args.no_save:
        save_post(args.topic, posts)


if __name__ == "__main__":
    main()
