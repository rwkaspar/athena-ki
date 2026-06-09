#!/usr/bin/env python3
"""Athena — Topic-Aliase in den ChromaDB-Metadaten konsolidieren.

Hintergrund: Das Auto-Review-LLM hat über die Zeit drei Schreibweisen für
Außenpolitik produziert (aussenpolitik, aussenspolitik, außenpolitik). Die
Quellen-Seite zeigt sie deshalb als drei separate Tags im Filter-Dropdown.
auto_review.py normalisiert ab jetzt vor dem Schreiben — dieses Skript fixt
den Bestand in ChromaDB rückwirkend.

Aufruf:
    python scripts/normalize_topics.py --dry-run    # nur zeigen, was sich ändern würde
    python scripts/normalize_topics.py              # tatsächlich umschreiben

Server-Modus (athena-chroma:8001) wird automatisch erkannt (siehe retrieval.py).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Quelle der Wahrheit: identisch zu TAG_ALIASES in auto_review.py.
# Wenn das Mapping wächst, hier mit-aktualisieren.
TAG_ALIASES = {
    "aussenpolitik": "außenpolitik",
    "aussenspolitik": "außenpolitik",
    "buergergeld": "bürgergeld",
}


def _normalize(topics_csv: str) -> str:
    """Normalisiert einen kommaseparierten Topic-String."""
    if not topics_csv:
        return topics_csv
    seen, out = set(), []
    for raw in topics_csv.split(","):
        t = raw.strip()
        if not t:
            continue
        canon = TAG_ALIASES.get(t.lower(), t)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return ",".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Nur zeigen, was sich ändern würde — nichts schreiben.")
    p.add_argument("--scope", default="bund",
                   help="Scope (bund oder pfofeld). Bestimmt die Collections.")
    args = p.parse_args()

    # Lokal importieren, damit das Skript auch ohne Chroma-Verbindung lädt
    # (z. B. zum Lesen der Docstring per --help).
    from retrieval import chroma_server_mode, get_chroma_client, collection_names_for

    server_mode = chroma_server_mode()
    print(f"[mode] {'Server (athena-chroma:8001)' if server_mode else 'Embedded (athena-db/)'}")

    client = get_chroma_client() if server_mode else None
    if client is None:
        # Embedded-Fallback: chromadb.PersistentClient
        import chromadb
        from ingest import CHROMA_DB_DIR
        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    try:
        collection_names = collection_names_for(args.scope)
    except Exception:
        # Fallback: alle Collections ablaufen, wenn die Helper-Funktion fehlt.
        collection_names = [c.name for c in client.list_collections()]

    total_chunks, changed_chunks = 0, 0
    by_collection = {}
    for name in collection_names:
        try:
            coll = client.get_collection(name=name)
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}")
            continue

        offset, page = 0, 5000
        coll_changed = 0
        while True:
            try:
                data = coll.get(include=["metadatas"], limit=page, offset=offset)
            except Exception as e:
                print(f"  [err]  {name} offset={offset}: {type(e).__name__}: {e}")
                break
            ids = data.get("ids") or []
            metas = data.get("metadatas") or []
            if not ids:
                break
            total_chunks += len(ids)
            # Sammle Chunks, die geändert werden müssen, in Batches.
            upd_ids, upd_metas = [], []
            for cid, meta in zip(ids, metas):
                raw = (meta or {}).get("topics") or ""
                norm = _normalize(raw)
                if norm != raw:
                    new_meta = dict(meta or {})
                    new_meta["topics"] = norm
                    upd_ids.append(cid)
                    upd_metas.append(new_meta)
            if upd_ids:
                coll_changed += len(upd_ids)
                changed_chunks += len(upd_ids)
                if args.dry_run:
                    # Zeige bis zu 3 Beispiele pro Batch.
                    for cid, meta in list(zip(upd_ids, upd_metas))[:3]:
                        print(f"  [diff] {name} {cid}: topics -> '{meta['topics']}'")
                else:
                    coll.update(ids=upd_ids, metadatas=upd_metas)
                    print(f"  [upd]  {name} offset={offset}: {len(upd_ids)} chunks normalisiert")
            if len(ids) < page:
                break
            offset += len(ids)
        by_collection[name] = coll_changed

    print(f"\n[done] Chunks gelesen: {total_chunks}, betroffen: {changed_chunks}")
    for n, c in by_collection.items():
        if c:
            print(f"  {n}: {c}")
    if args.dry_run:
        print("(dry-run — keine Schreibzugriffe)")


if __name__ == "__main__":
    main()
