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
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import html2text
import requests as req
import yaml
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from retrieval import collection_for_source_type

# Konfiguration
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
SOURCE_TIERS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "source_tiers.yaml"
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_embeddings():
    """Embedding-Modell initialisieren."""
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)


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


def load_source_tiers():
    """Quellen-Hierarchie aus YAML laden. Bei Fehler: nur Default-Tier."""
    fallback = {"tiers": [], "default": {"rank": 3, "label": "unclassified"}}
    try:
        with open(SOURCE_TIERS_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or "default" not in cfg:
            print(f"   ⚠️ {SOURCE_TIERS_PATH} ohne 'default' — Fallback aktiv")
            return fallback
        cfg.setdefault("tiers", [])
        return cfg
    except FileNotFoundError:
        print(f"   ⚠️ {SOURCE_TIERS_PATH} nicht gefunden — alle Quellen werden als Tier 3 klassifiziert")
        return fallback


def _domain_matches(netloc: str, registered: str) -> bool:
    netloc = (netloc or "").lower().lstrip(".")
    registered = registered.lower().lstrip(".")
    return netloc == registered or netloc.endswith("." + registered)


def classify_source(source: str, tiers_cfg: dict) -> tuple[str, int]:
    """Source (URL oder Pfad) zu (label, rank) klassifizieren.
    Pfade ohne URL-Schema → Default-Tier (Override per CLI)."""
    parsed = urlparse(source)
    if not parsed.scheme or parsed.scheme == "file":
        return tiers_cfg["default"]["label"], tiers_cfg["default"]["rank"]
    netloc = parsed.netloc
    for tier in tiers_cfg.get("tiers", []):
        for domain in tier.get("domains", []):
            if _domain_matches(netloc, domain):
                return tier["label"], tier["rank"]
    return tiers_cfg["default"]["label"], tiers_cfg["default"]["rank"]


def enrich_metadata(docs, tiers_cfg, override_rank=None, override_label=None):
    """Tier-Metadaten und Ingest-Zeitstempel auf Dokument-Ebene anhängen.
    Override greift für alle übergebenen Dokumente — z. B. wenn ein PDF
    explizit als Tier 1 markiert werden soll."""
    now = datetime.now(timezone.utc).isoformat()
    for doc in docs:
        source = doc.metadata.get("source", "")
        if override_rank is not None:
            label = override_label or "manual_override"
            rank = override_rank
        else:
            label, rank = classify_source(source, tiers_cfg)
        doc.metadata.update({
            "tier_rank": rank,
            "tier_label": label,
            "source_type": "static" if rank == 1 else "fresh",
            "ingested_at": now,
        })
    return docs


USER_AGENT = "Athena-KI/1.0 (Gemeinde Pfofeld; +https://github.com/rwkaspar/athena-ki)"
MAX_RETRIES = 3


def ingest_url(url: str):
    """Webseite laden und einspeisen."""
    print(f"📥 Lade URL: {url}")
    session = req.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    loader = WebBaseLoader(url, requests_per_second=1)
    loader.session = session
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            docs = loader.load()
            print(f"   {len(docs)} Dokument(e) geladen")
            return docs
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 2
                print(f"   ⚠️ Versuch {attempt} fehlgeschlagen, warte {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ Fehlgeschlagen nach {MAX_RETRIES} Versuchen: {e}")
                return []


def ingest_url_rendered(url: str, wait_selector: str | None = None):
    """Webseite via Headless-Chrome (Playwright) JS-gerendert laden.
    Nötig für Seiten wie gesetze-bayern.de, die Inhalte erst per JavaScript
    nachladen. wait_selector erlaubt das Warten auf einen konkreten DOM-Knoten,
    sonst wird auf 'networkidle' gewartet."""
    from playwright.sync_api import sync_playwright

    print(f"📥 Lade URL (JS-gerendert): {url}")
    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.body_width = 0  # kein hard-wrap

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=USER_AGENT)
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=60000)
                if wait_selector:
                    page.wait_for_selector(wait_selector, timeout=30000)
                html = page.content()
                title = page.title()
                browser.close()
            text = converter.handle(html).strip()
            print(f"   1 Dokument geladen, {len(text)} Zeichen Text")
            return [Document(
                page_content=text,
                metadata={"source": url, "title": title},
            )]
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 2
                print(f"   ⚠️ Versuch {attempt} fehlgeschlagen, warte {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ Fehlgeschlagen nach {MAX_RETRIES} Versuchen: {e}")
                return []


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


def _summarize_tiers(docs):
    """Verteilung der Tier-Labels für Transparenz im Log."""
    counts = {}
    for doc in docs:
        key = (doc.metadata.get("tier_rank", "?"), doc.metadata.get("tier_label", "?"))
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items())


def main():
    parser = argparse.ArgumentParser(
        description="Athena - Dokumente in die Wissensbasis einspeisen"
    )
    parser.add_argument("--url", help="URL einer Webseite")
    parser.add_argument("--file", help="Pfad zu einer Datei (PDF, TXT, MD)")
    parser.add_argument("--dir", help="Pfad zu einem Verzeichnis")
    parser.add_argument(
        "--render", action="store_true",
        help="URL via Headless-Browser laden (für JS-gerenderte Seiten wie gesetze-bayern.de)",
    )
    parser.add_argument(
        "--wait-selector",
        help="CSS-Selector, auf den nach dem Laden gewartet wird (nur mit --render)",
    )
    parser.add_argument(
        "--tier", type=int, choices=[1, 2, 3],
        help="Override-Tier für alle Dokumente dieses Aufrufs (1=Primär, 2=Medien, 3=Kommentar)",
    )
    parser.add_argument(
        "--source-label",
        help="Override-Label für alle Dokumente dieses Aufrufs (z. B. 'BayGO PDF 2024')",
    )

    args = parser.parse_args()

    if not any([args.url, args.file, args.dir]):
        parser.print_help()
        sys.exit(1)

    if args.source_label and args.tier is None:
        print("❌ --source-label nur zusammen mit --tier nutzen.")
        sys.exit(1)

    print("📚 Lade Quellen-Hierarchie...")
    tiers_cfg = load_source_tiers()

    # Dokumente laden
    docs = []
    if args.url:
        if args.render:
            docs.extend(ingest_url_rendered(args.url, wait_selector=args.wait_selector))
        else:
            docs.extend(ingest_url(args.url))
    if args.file:
        docs.extend(ingest_file(args.file))
    if args.dir:
        docs.extend(ingest_directory(args.dir))

    if not docs:
        print("❌ Keine Dokumente geladen.")
        sys.exit(1)

    # Tier-Metadaten anreichern
    docs = enrich_metadata(
        docs, tiers_cfg,
        override_rank=args.tier,
        override_label=args.source_label,
    )
    print("\n🏷️  Tier-Verteilung:")
    for (rank, label), n in _summarize_tiers(docs):
        print(f"   Tier {rank} / {label}: {n} Dokument(e)")

    # In Chunks aufteilen
    print(f"\n✂️  Teile {len(docs)} Dokument(e) in Chunks auf...")
    chunks = split_documents(docs)
    print(f"   {len(chunks)} Chunks erstellt")

    # Chunks pro Ziel-Collection (static/fresh) gruppieren
    by_collection: dict[str, list] = {}
    for chunk in chunks:
        target = collection_for_source_type(chunk.metadata.get("source_type", "fresh"))
        by_collection.setdefault(target, []).append(chunk)

    # Embeddings erstellen und in die jeweilige Collection schreiben
    print(f"\n💾 Speichere in ChromaDB ({CHROMA_DB_DIR})...")
    embeddings = get_embeddings()
    for collection_name, batch in by_collection.items():
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
        )
        vectorstore.add_documents(batch)
        print(f"   ✅ {len(batch)} Chunks → Collection '{collection_name}'")
    print(f"   Datenbank: {CHROMA_DB_DIR}")


if __name__ == "__main__":
    main()
