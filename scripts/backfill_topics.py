#!/usr/bin/env python3
"""Athena — Themen-Tags für bestehende Quellen nachtragen (Backfill).

Geht durch alle Quellen eines Scopes in ChromaDB, lässt Mistral anhand von
Titel + Textauszug Themen-Tags vergeben (Leitplanken-Liste gegen Wildwuchs)
und schreibt sie als 'topics'-Metadatum (kommasepariert) in alle Chunks der
Quelle. Überspringt Quellen, die schon Tags haben (resumierbar).

Aufruf:
    python scripts/backfill_topics.py --scope bund [--limit N] [--dry-run]
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from retrieval import collection_names_for

CHROMA_DB_DIR = Path(__file__).parent.parent / "athena-db"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
REVIEW_MODEL = os.getenv("ATHENA_REVIEW_MODEL", "mistral-large-latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TAG_MODEL = os.getenv("ATHENA_TAG_MODEL", "qwen3.5:latest")

# gleiche Leitplanken wie auto_review (Konsistenz)
from auto_review import SUGGESTED_TAGS


def _build_llm(provider: str):
    """provider='mistral' (API, rate-limited) oder 'ollama' (lokal auf aitest,
    kein Limit, blockiert den Mistral-Chat-Key nicht). Default ollama."""
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        if not MISTRAL_API_KEY:
            raise RuntimeError("MISTRAL_API_KEY nicht gesetzt.")
        return ChatMistralAI(model=REVIEW_MODEL, api_key=MISTRAL_API_KEY,
                             temperature=0.1, max_retries=2, timeout=90)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_TAG_MODEL, base_url=OLLAMA_HOST,
                      temperature=0.1, timeout=120, reasoning=False, num_gpu=0)


def classify(llm, title: str, sample: str) -> list[str]:
    prompt = f"""Vergib 2-5 Themen-Tags für diese Quelle einer deutschen politischen Wissensbasis.
Wähle BEVORZUGT aus dieser Liste (neue nur wenn nötig): {', '.join(SUGGESTED_TAGS)}

Antworte NUR mit einem JSON-Array von Strings, z.B. ["klima","energie"].

TITEL: {title}
AUSZUG: {sample[:2500]}"""
    raw = llm.invoke(prompt)
    content = raw.content if hasattr(raw, "content") else str(raw)
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if not m:
        return []
    try:
        tags = json.loads(m.group(0))
        return [str(t).strip().lower() for t in tags if str(t).strip()][:5]
    except Exception:
        return []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", default="bund", choices=["pfofeld", "bund"])
    p.add_argument("--limit", type=int, default=0, help="max. N Quellen (0=alle)")
    p.add_argument("--provider", default="ollama", choices=["ollama", "mistral"],
                   help="ollama=lokal/kein Limit (Default), mistral=API/rate-limited")
    p.add_argument("--delay", type=float, default=0.0, help="Pause zwischen LLM-Calls (s); für mistral ~2-4 gegen Rate-Limit")
    p.add_argument("--dry-run", action="store_true", help="nur anzeigen, nicht schreiben")
    args = p.parse_args()

    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    llm = None if args.dry_run else _build_llm(args.provider)

    # Quellen pro Collection sammeln: source → (chunk_ids, sample, title, hat_topics)
    done = 0
    for coll_name in collection_names_for(args.scope):
        try:
            coll = client.get_collection(coll_name)
        except Exception:
            continue
        data = coll.get()  # alle chunks
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        docs = data.get("documents") or []

        # nach source gruppieren
        by_source: dict[str, dict] = {}
        for i, m in enumerate(metas):
            src = (m or {}).get("source") or "?"
            e = by_source.setdefault(src, {"ids": [], "title": (m or {}).get("title") or src,
                                           "sample": "", "has_topics": False})
            e["ids"].append(ids[i])
            if (m or {}).get("topics"):
                e["has_topics"] = True
            if len(e["sample"]) < 2500 and i < len(docs):
                e["sample"] += " " + (docs[i] or "")

        for src, e in by_source.items():
            if e["has_topics"]:
                continue  # schon getaggt → skip (resumierbar)
            if args.limit and done >= args.limit:
                break
            if args.dry_run:
                print(f"[dry] würde taggen: {e['title'][:60]}  ({len(e['ids'])} chunks)")
                done += 1
                continue
            # Retry mit Backoff bei Rate-Limit (429)
            tags = []
            for attempt in range(4):
                try:
                    tags = classify(llm, e["title"], e["sample"])
                    break
                except Exception as ex:
                    if "429" in str(ex) or "rate" in str(ex).lower():
                        wait = args.delay * (attempt + 1) * 4
                        print(f"[wait] Rate-Limit, warte {wait:.0f}s …", file=sys.stderr)
                        time.sleep(wait)
                    else:
                        print(f"[err] {e['title'][:40]}: {type(ex).__name__}", file=sys.stderr)
                        break
            if not tags:
                print(f"[skip] keine Tags für {e['title'][:50]}", file=sys.stderr)
                continue
            topics_str = ",".join(tags)
            # alle chunks dieser Quelle updaten
            cur = coll.get(ids=e["ids"])
            new_metas = [m or {} for m in (cur.get("metadatas") or [])]
            for m in new_metas:
                m["topics"] = topics_str
            coll.update(ids=e["ids"], metadatas=new_metas)
            done += 1
            print(f"[ok] {e['title'][:55]} → {topics_str}")
            time.sleep(args.delay)  # Drosselung gegen Rate-Limit

    print(f"\nFertig. {done} Quellen getaggt{' (dry-run)' if args.dry_run else ''}.")


if __name__ == "__main__":
    main()
