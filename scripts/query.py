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

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA

# Konfiguration
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "athena"


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


def ask(qa_chain, question: str):
    """Eine Frage stellen und Antwort mit Quellen ausgeben."""
    print(f"\n🔍 Frage: {question}")
    print("⏳ Athena denkt nach...\n")
    start_time = time.time()

    result = qa_chain.invoke(question)

    print("=" * 60)
    print("🏛️  ATHENA:")
    print("=" * 60)
    print(result["result"])
    print("\n" + "-" * 60)
    elapsed = time.time() - start_time
    print(f"\n⏱️  Antwortzeit: {elapsed:.1f} Sekunden")
    print("\n" + "-" * 60)
    print("📚 Quellen:")
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "Unbekannt")
        print(f"   [{i}] {source}")
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

    print("🔄 Lade Athena...")
    qa_chain = create_qa_chain()

    if args.interactive:
        interactive_mode(qa_chain)
    else:
        ask(qa_chain, args.question)


if __name__ == "__main__":
    main()
