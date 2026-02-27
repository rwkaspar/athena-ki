#!/usr/bin/env python3
"""
Athena - Dokumente in die Wissensbasis einspeisen

Verwendung:
    python ingest.py --url https://www.gesetze-im-internet.de/gg/
    python ingest.py --file ../documents/gesetze/grundgesetz.pdf
    python ingest.py --dir ../documents/gesetze/
"""

import argparse
import os
import sys
from pathlib import Path

from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Konfiguration
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_embeddings():
    """Embedding-Modell initialisieren."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore(embeddings):
    """ChromaDB Vektordatenbank laden oder erstellen."""
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )


def split_documents(docs):
    """Dokumente in Chunks aufteilen."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def ingest_url(url: str):
    """Webseite laden und einspeisen."""
    print(f"📥 Lade URL: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    print(f"   {len(docs)} Dokument(e) geladen")
    return docs


def ingest_file(filepath: str):
    """Einzelne Datei laden und einspeisen."""
    path = Path(filepath)
    print(f"📥 Lade Datei: {path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() in (".txt", ".md"):
        loader = TextLoader(str(path))
    else:
        print(f"   ⚠️ Nicht unterstütztes Format: {path.suffix}")
        return []

    docs = loader.load()
    print(f"   {len(docs)} Dokument(e) geladen")
    return docs


def ingest_directory(dirpath: str):
    """Alle Dateien in einem Verzeichnis laden."""
    print(f"📥 Lade Verzeichnis: {dirpath}")
    docs = []

    for path in Path(dirpath).rglob("*"):
        if path.is_file() and path.suffix.lower() in (".pdf", ".txt", ".md"):
            docs.extend(ingest_file(str(path)))

    print(f"   Gesamt: {len(docs)} Dokument(e) geladen")
    return docs


def main():
    parser = argparse.ArgumentParser(
        description="Athena - Dokumente in die Wissensbasis einspeisen"
    )
    parser.add_argument("--url", help="URL einer Webseite")
    parser.add_argument("--file", help="Pfad zu einer Datei (PDF, TXT, MD)")
    parser.add_argument("--dir", help="Pfad zu einem Verzeichnis")

    args = parser.parse_args()

    if not any([args.url, args.file, args.dir]):
        parser.print_help()
        sys.exit(1)

    # Dokumente laden
    docs = []
    if args.url:
        docs.extend(ingest_url(args.url))
    if args.file:
        docs.extend(ingest_file(args.file))
    if args.dir:
        docs.extend(ingest_directory(args.dir))

    if not docs:
        print("❌ Keine Dokumente geladen.")
        sys.exit(1)

    # In Chunks aufteilen
    print(f"\n✂️  Teile {len(docs)} Dokument(e) in Chunks auf...")
    chunks = split_documents(docs)
    print(f"   {len(chunks)} Chunks erstellt")

    # Embeddings erstellen und speichern
    print(f"\n💾 Speichere in ChromaDB ({CHROMA_DB_DIR})...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"✅ {len(chunks)} Chunks erfolgreich eingespeist!")
    print(f"   Datenbank: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
