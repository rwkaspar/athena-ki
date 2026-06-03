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
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _tiers_path_for(scope: str) -> str:
    """Pfad zur Tier-YAML für einen Scope. Pfofeld ohne Suffix (Default-Datei),
    andere Scopes mit Suffix: config/source_tiers_<scope>.yaml."""
    if scope == "pfofeld":
        return os.path.join(CONFIG_DIR, "source_tiers.yaml")
    return os.path.join(CONFIG_DIR, f"source_tiers_{scope}.yaml")


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


def load_source_tiers(scope: str = "pfofeld"):
    """Quellen-Hierarchie aus YAML laden. Bei Fehler: nur Default-Tier."""
    fallback = {"tiers": [], "default": {"rank": 3, "label": "unclassified"}}
    path = _tiers_path_for(scope)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict) or "default" not in cfg:
            print(f"   ⚠️ {path} ohne 'default' — Fallback aktiv")
            return fallback
        cfg.setdefault("tiers", [])
        return cfg
    except FileNotFoundError:
        print(f"   ⚠️ {path} nicht gefunden — alle Quellen werden als Tier 3 klassifiziert")
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


def enrich_metadata(docs, tiers_cfg, override_rank=None, override_label=None, topics=None):
    """Tier-Metadaten und Ingest-Zeitstempel auf Dokument-Ebene anhängen.
    Override greift für alle übergebenen Dokumente — z. B. wenn ein PDF
    explizit als Tier 1 markiert werden soll.

    topics: optionaler kommaseparierter String von Themen-Tags. ChromaDB-
    Metadaten können keine Listen speichern, daher als String abgelegt und
    beim Filtern geparst (siehe /sources)."""
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
        if topics:
            doc.metadata["topics"] = topics
    return docs


USER_AGENT = "Athena-KI/1.0 (Gemeinde Pfofeld; +https://github.com/rwkaspar/athena-ki)"
MAX_RETRIES = 3
URL_TIMEOUT_S = 60          # Timeout für HTML-Loads
PDF_DOWNLOAD_TIMEOUT_S = 120  # PDFs koennen groesser sein
MIN_RAW_TEXT_LEN = 200      # weniger als das gilt als "leer" -> Fallback


def _looks_like_pdf(url: str) -> bool:
    """Heuristik: URL endet auf .pdf (mit oder ohne Query)."""
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def ingest_pdf_url(url: str):
    """PDF herunterladen (mit Timeout) und ueber PyPDFLoader parsen.
    Vermeidet das Haengen von WebBaseLoader bei PDF-URLs."""
    print(f"📥 Lade PDF: {url}")
    import tempfile
    headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = req.get(url, headers=headers, timeout=PDF_DOWNLOAD_TIMEOUT_S, stream=True, allow_redirects=True)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "").lower()
            if "pdf" not in ctype and "application/octet-stream" not in ctype:
                print(f"   ⚠️ Content-Type ist '{ctype}', kein PDF")
                # nicht abbrechen — manche Server senden text/html und liefern trotzdem PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                for chunk in r.iter_content(chunk_size=64*1024):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
            size = os.path.getsize(tmp_path)
            print(f"   PDF geladen: {size} Bytes -> {tmp_path}")
            if size < 1024:
                print(f"   ⚠️ PDF verdaechtig klein, ueberspringe")
                os.unlink(tmp_path)
                return []
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                # source-Metadaten auf Original-URL setzen, nicht Temp-Pfad
                for d in docs:
                    d.metadata["source"] = url
                print(f"   {len(docs)} Seite(n) geladen aus PDF")
                return docs
            finally:
                os.unlink(tmp_path)
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 2
                print(f"   ⚠️ Versuch {attempt} fehlgeschlagen: {type(e).__name__}: {e}, warte {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ Fehlgeschlagen nach {MAX_RETRIES} Versuchen: {e}")
                return []


def ingest_url(url: str, *, allow_render_fallback: bool = True):
    """Dispatcher: PDF -> ingest_pdf_url, sonst HTML via WebBaseLoader mit
    Timeout, bei leerem Ergebnis automatisch Fallback auf ingest_url_rendered."""
    if _looks_like_pdf(url):
        return ingest_pdf_url(url)

    print(f"📥 Lade URL: {url}")
    session = req.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    loader = WebBaseLoader(
        url,
        requests_per_second=1,
        requests_kwargs={"timeout": URL_TIMEOUT_S},
    )
    loader.session = session
    docs = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            docs = loader.load()
            print(f"   {len(docs)} Dokument(e) geladen")
            break
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 2
                print(f"   ⚠️ Versuch {attempt} fehlgeschlagen ({type(e).__name__}: {e}), warte {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ Fehlgeschlagen nach {MAX_RETRIES} Versuchen: {e}")
                docs = []

    # Auto-Render-Fallback wenn Ergebnis leer/duenn
    total_chars = sum(len((d.page_content or "").strip()) for d in docs)
    if allow_render_fallback and total_chars < MIN_RAW_TEXT_LEN:
        print(f"   ↪ HTML-Lader lieferte nur {total_chars} Zeichen, versuche JS-Render-Fallback...")
        try:
            rendered = ingest_url_rendered(url)
            rendered_chars = sum(len((d.page_content or "").strip()) for d in rendered)
            if rendered_chars > total_chars:
                print(f"   ✓ Render-Fallback lieferte {rendered_chars} Zeichen statt {total_chars}")
                return rendered
        except Exception as e:
            print(f"   ⚠️ Render-Fallback fehlgeschlagen: {type(e).__name__}: {e}")
    return docs


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
    parser.add_argument(
        "--scope", choices=["pfofeld", "bund"], default="pfofeld",
        help="Wissensbasis-Scope. pfofeld = Default (existierende Pilot-Collections); bund = Bundes-Collections",
    )
    parser.add_argument(
        "--topics",
        help="Kommaseparierte Themen-Tags für alle Dokumente dieses Aufrufs (z. B. 'klima,energie,eu-recht')",
    )

    args = parser.parse_args()

    if not any([args.url, args.file, args.dir]):
        parser.print_help()
        sys.exit(1)

    if args.source_label and args.tier is None:
        print("❌ --source-label nur zusammen mit --tier nutzen.")
        sys.exit(1)

    print(f"📚 Lade Quellen-Hierarchie (scope={args.scope})...")
    tiers_cfg = load_source_tiers(args.scope)

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
        topics=args.topics,
    )
    print("\n🏷️  Tier-Verteilung:")
    for (rank, label), n in _summarize_tiers(docs):
        print(f"   Tier {rank} / {label}: {n} Dokument(e)")

    # In Chunks aufteilen
    print(f"\n✂️  Teile {len(docs)} Dokument(e) in Chunks auf...")
    chunks = split_documents(docs)
    print(f"   {len(chunks)} Chunks erstellt")

    # Chunks pro Ziel-Collection (static/fresh) gruppieren — scope-aware
    by_collection: dict[str, list] = {}
    for chunk in chunks:
        target = collection_for_source_type(
            chunk.metadata.get("source_type", "fresh"), args.scope
        )
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
