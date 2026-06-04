#!/usr/bin/env python3
"""Recovery für die beschädigte ChromaDB-Collection `bund_fresh`.

Hintergrund: Paralleler Multi-Prozess-Schreibzugriff (Bulk-Ingest neben laufendem
uvicorn) hat den hnswlib-Vektorindex von `bund_fresh` zerstört — schon count()
segfaultet. SQLite ist intakt: Texte + Metadaten aller Chunks sind vorhanden.

Strategie:
  extract : liest Texte+Metadaten der bund_fresh-Chunks read-only aus SQLite und
            sichert sie als JSONL (kein ChromaDB-Zugriff → crashsicher).
  rebuild : löscht die kaputte Collection, baut sie neu auf und bettet die Texte
            über OllamaEmbeddings neu ein (keine PDFs erneut laden).

WICHTIG: Vor rebuild muss uvicorn GESTOPPT sein (sonst erneuter Parallelzugriff).

Aufruf:
    python scripts/recover_bund_fresh.py extract
    python scripts/recover_bund_fresh.py rebuild [--batch 200]
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent
DB = HERE.parent / "athena-db" / "chroma.sqlite3"
DB_DIR = HERE.parent / "athena-db"
DUMP = Path("/tmp/bund_fresh_recovery.jsonl")
COLLECTION = "bund_fresh"
# Metadata-Segment der bund_fresh-Collection (aus segments-Tabelle)
META_SEGMENT = "79ee68dd-bd61-4715-b072-119184b07eaf"
DOC_KEY = "chroma:document"

sys.path.insert(0, str(HERE))


def _value(string_value, int_value, float_value, bool_value):
    if string_value is not None:
        return string_value
    if int_value is not None:
        return int_value
    if float_value is not None:
        return float_value
    if bool_value is not None:
        return bool(bool_value)
    return None


def extract():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = con.cursor()
    # Alle internen ids + chroma-uuid der bund_fresh-Chunks
    rows = cur.execute(
        "SELECT id, embedding_id FROM embeddings WHERE segment_id=?", (META_SEGMENT,)
    ).fetchall()
    print(f"[extract] {len(rows)} Chunks im bund_fresh-Segment", file=sys.stderr)

    n_ok = n_skip = 0
    with DUMP.open("w", encoding="utf-8") as out:
        for internal_id, uuid in rows:
            md_rows = cur.execute(
                "SELECT key, string_value, int_value, float_value, bool_value "
                "FROM embedding_metadata WHERE id=?", (internal_id,)
            ).fetchall()
            doc = None
            meta = {}
            for key, sv, iv, fv, bv in md_rows:
                val = _value(sv, iv, fv, bv)
                if key == DOC_KEY:
                    doc = val
                elif val is not None:
                    meta[key] = val
            if not doc:
                n_skip += 1
                continue
            out.write(json.dumps({"id": uuid, "document": doc, "metadata": meta},
                                  ensure_ascii=False) + "\n")
            n_ok += 1
    con.close()
    print(f"[extract] gesichert: {n_ok} Chunks → {DUMP} (ohne Text übersprungen: {n_skip})")


def rebuild(batch_size):
    if not DUMP.exists():
        sys.exit("[rebuild] Kein Dump gefunden — erst 'extract' laufen lassen.")
    # Sicherheit: uvicorn darf nicht laufen
    import subprocess
    try:
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "athena-uvicorn.service"],
            capture_output=True, text=True,
            env={**os.environ, "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"},
        ).stdout.strip()
        if active == "active":
            sys.exit("[rebuild] ABBRUCH: uvicorn läuft. Erst stoppen "
                     "(systemctl --user stop athena-uvicorn.service).")
    except FileNotFoundError:
        pass

    # Datei-Iteration trennt nur an '\n' — splitlines() würde auch an U+2028/U+2029
    # u.ä. trennen (kommen in PDF-Texten vor, json.dumps escaped sie nicht).
    records = []
    with DUMP.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[rebuild] {len(records)} Chunks aus Dump geladen", file=sys.stderr)

    import serve
    emb = serve._get_embeddings()

    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Kaputte Collection entfernen
    print("[rebuild] lösche beschädigte Collection bund_fresh …", file=sys.stderr)
    try:
        client.delete_collection(COLLECTION)
        print("[rebuild] gelöscht.", file=sys.stderr)
    except Exception as e:
        print(f"[rebuild] delete_collection-Fehler ({type(e).__name__}: {e}) — "
              "ggf. schon weg, fahre fort.", file=sys.stderr)

    from langchain_chroma import Chroma
    vs = Chroma(collection_name=COLLECTION, embedding_function=emb,
                persist_directory=str(DB_DIR))

    total = len(records)
    done = 0
    for i in range(0, total, batch_size):
        chunk = records[i:i + batch_size]
        vs.add_texts(
            texts=[r["document"] for r in chunk],
            metadatas=[r["metadata"] for r in chunk],
            ids=[r["id"] for r in chunk],
        )
        done += len(chunk)
        print(f"[rebuild] {done}/{total} eingebettet", file=sys.stderr)

    cnt = vs._collection.count()
    print(f"[rebuild] FERTIG. bund_fresh count={cnt}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    rb = sub.add_parser("rebuild")
    rb.add_argument("--batch", type=int, default=200)
    args = p.parse_args()
    if args.cmd == "extract":
        extract()
    else:
        rebuild(args.batch)


if __name__ == "__main__":
    main()
