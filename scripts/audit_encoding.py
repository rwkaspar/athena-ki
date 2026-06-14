#!/usr/bin/env python3
"""Athena — Bestands-Bereinigung Encoding-Probleme.

Iteriert alle Chunks und repariert über text_clean.clean_text():
  * PUA-Reste aus kaputten PDF-Schriften
  * Replacement-Chars „�"
  * Ligaturen (ﬁ ﬂ ﬃ ﬄ)
  * Soft-Hyphens, Zero-Width, BOM mitte
  * Mojibake (UTF-8-als-Latin1)

ChromaDB.update(documents=…) embedded automatisch neu — das ist teuer
(GPU/CPU + Ollama-Rate), deshalb lohnt sich der Aufwand nur, wenn die
Chunks wirklich Probleme haben (`needs_cleaning`).

Aufruf:
    python scripts/audit_encoding.py --dry-run    # nur zählen + Beispiele
    python scripts/audit_encoding.py              # umschreiben + re-embedden
    python scripts/audit_encoding.py --no-reembed # NUR documents updaten ohne Embeddings (riskant)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Nur zeigen, was sich ändern würde.")
    p.add_argument("--scope", default="bund", help="bund oder pfofeld.")
    p.add_argument("--limit-per-batch", type=int, default=200,
                   help="Re-Embedding-Batchgröße (gegen Ollama-Stress).")
    p.add_argument("--no-reembed", action="store_true",
                   help="WARNUNG: nur documents updaten, Embeddings bleiben stale.")
    p.add_argument("--source", help="Nur eine bestimmte source-URL bearbeiten.")
    args = p.parse_args()

    from text_clean import clean_text, needs_cleaning
    from retrieval import get_chroma_client, collection_names_for

    # WICHTIG: ChromaDB's `update(documents=...)` würde sonst die Collection-eigene
    # Embedding-Funktion benutzen — die ist auf das ONNX-Default-Modell gesetzt,
    # nicht auf unser bge-m3. Wir berechnen Embeddings selbst und übergeben sie.
    embedder = None
    if not args.no_reembed and not args.dry_run:
        from langchain_ollama import OllamaEmbeddings
        embedder = OllamaEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
        print("[embedder] bge-m3 via Ollama bereit", file=sys.stderr)

    client = get_chroma_client()
    total, dirty, fixed = 0, 0, 0
    examples = []

    for name in collection_names_for(args.scope):
        try:
            coll = client.get_collection(name=name)
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue

        offset = 0
        upd_ids, upd_docs = [], []
        while True:
            try:
                where = {"source": args.source} if args.source else None
                kw = {"include": ["metadatas", "documents"], "limit": 2000, "offset": offset}
                if where:
                    kw["where"] = where
                data = coll.get(**kw)
            except Exception as e:
                print(f"  [err] {name} offset={offset}: {type(e).__name__}: {e}", file=sys.stderr)
                break
            ids = data.get("ids") or []
            metas = data.get("metadatas") or []
            docs = data.get("documents") or []
            if not ids:
                break
            for cid, meta, doc in zip(ids, metas, docs):
                total += 1
                if not needs_cleaning(doc):
                    continue
                dirty += 1
                cleaned = clean_text(doc)
                if cleaned == doc:
                    continue
                upd_ids.append(cid)
                upd_docs.append(cleaned)
                if len(examples) < 3:
                    examples.append((cid, doc[:120], cleaned[:120]))

            # In handhabbaren Batches re-embedden, damit Ollama nicht overflowed.
            while len(upd_ids) >= args.limit_per_batch:
                batch_ids = upd_ids[:args.limit_per_batch]
                batch_docs = upd_docs[:args.limit_per_batch]
                upd_ids = upd_ids[args.limit_per_batch:]
                upd_docs = upd_docs[args.limit_per_batch:]
                if args.dry_run:
                    print(f"  [dry] {name}: würde {len(batch_ids)} Chunks neu schreiben")
                elif args.no_reembed:
                    # NUR documents, KEINE Embeddings → ChromaDB würde
                    # eigene Default-Function aufrufen → ONNX-Download. Wir
                    # übergeben EXPLIZIT die alten Embeddings, damit nichts
                    # neu gerechnet wird.
                    old = coll.get(ids=batch_ids, include=["embeddings"])
                    coll.update(ids=batch_ids, documents=batch_docs,
                                embeddings=old.get("embeddings"))
                    fixed += len(batch_ids)
                    print(f"  [upd] {name}: {len(batch_ids)} Chunks (NO RE-EMBED!)")
                else:
                    new_emb = embedder.embed_documents(batch_docs)
                    coll.update(ids=batch_ids, documents=batch_docs, embeddings=new_emb)
                    fixed += len(batch_ids)
                    print(f"  [upd] {name}: {len(batch_ids)} Chunks (re-embedded)")
            if len(ids) < 2000:
                break
            offset += len(ids)

        # Rest-Batch
        if upd_ids:
            if args.dry_run:
                print(f"  [dry] {name}: würde {len(upd_ids)} Chunks neu schreiben")
            elif args.no_reembed:
                old = coll.get(ids=upd_ids, include=["embeddings"])
                coll.update(ids=upd_ids, documents=upd_docs,
                            embeddings=old.get("embeddings"))
                fixed += len(upd_ids)
                print(f"  [upd] {name}: {len(upd_ids)} Chunks (NO RE-EMBED!)")
            else:
                new_emb = embedder.embed_documents(upd_docs)
                coll.update(ids=upd_ids, documents=upd_docs, embeddings=new_emb)
                fixed += len(upd_ids)
                print(f"  [upd] {name}: {len(upd_ids)} Chunks (re-embedded)")

    print(f"\n[done] Chunks gelesen: {total}, "
          f"erkannt: {dirty}, {'würden geändert' if args.dry_run else 'geändert'}: "
          f"{dirty if args.dry_run else fixed}")
    if examples:
        print("\nBeispiele (gekürzt):")
        for cid, before, after in examples:
            print(f"  {cid}")
            print(f"    vor : {before!r}")
            print(f"    nach: {after!r}")


if __name__ == "__main__":
    main()
