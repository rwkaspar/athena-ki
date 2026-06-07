#!/usr/bin/env python3
"""Backfill lesbarer Titel für Quellen ohne `title` in der ChromaDB.

Idee: Den Titel aus dem ERSTEN gespeicherten Chunk der Quelle ableiten (der
enthält fast immer den Dokumentanfang) — kein Re-Download, kein 429-Risiko.
Kuratierte OVERRIDES haben Vorrang für Schlüsselquellen.

Setzt zwei Felder:
  title       — kurzer Anzeigetitel (für die Quellen-Zeile)
  title_full  — vollständiger Originaltitel (Hover/Tooltip)

Default: DRY-RUN (zeigt nur, was es täte). Mit --apply wird geschrieben.
WICHTIG: ChromaDB verträgt keine parallelen Writes → uvicorn vorher stoppen.
Der --apply-Pfad bricht ab, wenn athena-uvicorn aktiv ist (Override: --force).
"""
import os, sys, re, subprocess
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import chromadb
from retrieval import get_chroma_client, chroma_server_mode

DB = os.path.join(os.path.dirname(__file__), "..", "athena-db")
APPLY = "--apply" in sys.argv
FORCE = "--force" in sys.argv

# Kuratierte Titel für Schlüsselquellen (Vorrang vor Heuristik).
OVERRIDES = {
    "https://eur-lex.europa.eu/legal-content/DE/TXT/PDF/?uri=OJ:L_202401689": {
        "title": "Verordnung (EU) 2024/1689 – KI-Verordnung (AI Act)",
        "title_full": (
            "VERORDNUNG (EU) 2024/1689 DES EUROPÄISCHEN PARLAMENTS UND DES RATES "
            "vom 13. Juni 2024 zur Festlegung harmonisierter Vorschriften für "
            "künstliche Intelligenz und zur Änderung der Verordnungen (EG) Nr. 300/2008, "
            "(EU) Nr. 167/2013, (EU) Nr. 168/2013, (EU) 2018/858, (EU) 2018/1139 und "
            "(EU) 2019/2144 sowie der Richtlinien 2014/90/EU, (EU) 2016/797 und "
            "(EU) 2020/1828 (Verordnung über künstliche Intelligenz)"
        ),
    },
}

_NOISE = re.compile(r"^(seite \d+|page \d+|\d+\s*/\s*\d+|amtsblatt|official journal|de\b|\W*)$", re.I)
_CITATION = re.compile(r"^[A-ZÄÖÜ][\wäöüß]+,\s+[A-ZÄÖÜ]")            # "Brüggemann, A" / "Marcks, Holger"
_CITE_YEAR = re.compile(r"\.\s+(19|20)\d\d[.:]")                      # "… Maik. 2022." (Zitatliste)
_NOISE_START = re.compile(r"^(anhang|signatur|tabelle|schaubild|anlage|quelle|abbildung|abb\.|vgl\.|bearbeiter|impressum|inhalt)\b", re.I)
_OCR_BREAK = re.compile(r"\w+-\s+\w")                                  # "verklei- nert"


def _looks_clean(s: str) -> bool:
    """True, wenn die Zeile wie ein echter Titel aussieht (kein OCR-Müll/Zitat/Fragment)."""
    if not s or not (8 <= len(s) <= 200):
        return False
    if s[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ":   # echte Titel: Latein-Großbuchstabe vorn
        return False
    if _CITATION.match(s) or _NOISE_START.match(s) or _OCR_BREAK.search(s) or _CITE_YEAR.search(s):
        return False
    # OCR-Müll: viele „Wörter" ohne Vokal (z.B. 'blelbt','vH','vaainbt')
    toks = [t for t in re.findall(r"[A-Za-zÄÖÜäöüß]{2,}", s)]
    if toks:
        novowel = sum(1 for t in toks if not re.search(r"[aeiouäöüAEIOUÄÖÜ]", t))
        if novowel / len(toks) > 0.15:
            return False
    return True


def derive_from_text(text: str):
    """Erste sinnvolle Zeile(n) als Titel. Liefert (title_kurz, title_full) oder (None,None)."""
    lines = [l.strip() for l in (text or "").splitlines()]
    lines = [l for l in lines if len(l) >= 6 and not _NOISE.match(l)]
    if not lines or not _looks_clean(lines[0]):
        return None, None
    full = lines[0]
    for l in lines[1:3]:
        if len(full) > 120:
            break
        full += " " + l
    full = re.sub(r"\s+", " ", full).strip()[:300]
    short = full[:78] + "…" if len(full) > 80 else full
    return short, full


def main():
    client = get_chroma_client()
    # source -> {ids_by_collection, current_title, first_text, first_idx}
    src = {}
    for cname in [c.name for c in client.list_collections()]:
        col = client.get_collection(cname)
        off = 0
        while True:
            try:
                got = col.get(include=["metadatas", "documents"], limit=4000, offset=off)
            except Exception as e:
                print(f"[warn] {cname} offset {off}: {e}", file=sys.stderr)
                break
            ids = got.get("ids") or []
            metas = got.get("metadatas") or []
            docs = got.get("documents") or []
            if not ids:
                break
            for i, m in zip(ids, metas):
                m = m or {}
                s = m.get("source")
                if not s:
                    continue
                e = src.setdefault(s, {"cols": {}, "title": m.get("title") or "",
                                       "first_text": "", "first_idx": 10**9})
                e["cols"].setdefault(cname, []).append(i)
                if m.get("title"):
                    e["title"] = m["title"]
            for i, m, d in zip(ids, metas, docs):
                m = m or {}
                s = m.get("source")
                if not s:
                    continue
                idx = m.get("chunk_index", 0) or 0
                if idx < src[s]["first_idx"]:
                    src[s]["first_idx"] = idx
                    src[s]["first_text"] = d or ""
            off += len(ids)
            if len(ids) < 4000:
                break

    untitled = {s: e for s, e in src.items() if not e["title"]}
    plan = []  # (source, title, title_full, n_chunks)
    for s, e in untitled.items():
        if s in OVERRIDES:
            t, tf = OVERRIDES[s]["title"], OVERRIDES[s]["title_full"]
        else:
            t, tf = derive_from_text(e["first_text"])
        if not t:
            continue
        n = sum(len(v) for v in e["cols"].values())
        plan.append((s, t, tf, n, e["cols"]))

    print(f"Quellen gesamt: {len(src)} | ohne Titel: {len(untitled)} | mit Titel-Vorschlag: {len(plan)}\n")
    for s, t, tf, n, _ in sorted(plan, key=lambda x: -x[3]):
        print(f"[{n:>4} chunks] {s[:70]}")
        print(f"            → title: {t}")
    if not APPLY:
        print("\n(DRY-RUN — nichts geschrieben. Mit --apply ausführen, uvicorn vorher stoppen.)")
        return

    # Schreibschutz nur im embedded-Modus — im Server-Modus sind parallele Writes sicher.
    if not chroma_server_mode():
        active = subprocess.run(["systemctl", "--user", "is-active", "athena-uvicorn"],
                                capture_output=True, text=True).stdout.strip()
        if active == "active" and not FORCE:
            print("\n[ABBRUCH] athena-uvicorn ist AKTIV — erst stoppen "
                  "(systemctl --user stop athena-uvicorn) oder --force.", file=sys.stderr)
            sys.exit(1)

    written = 0
    for cname in {c for _, _, _, _, cols in plan for c in cols}:
        col = client.get_collection(cname)
        for s, t, tf, n, cols in plan:
            ids = cols.get(cname)
            if not ids:
                continue
            # volle Metadaten je id holen, title/title_full ergänzen, zurückschreiben
            for b in range(0, len(ids), 500):   # batchen (große Quellen wie eur-lex: 3431 chunks)
                batch = ids[b:b + 500]
                got = col.get(ids=batch, include=["metadatas"])
                new_metas = []
                for m in got.get("metadatas") or []:
                    m = dict(m or {})
                    m["title"] = t
                    m["title_full"] = tf
                    new_metas.append(m)
                col.update(ids=got["ids"], metadatas=new_metas)
                written += len(got["ids"])
    print(f"\n[ok] {written} Chunks aktualisiert ({len(plan)} Quellen).")


if __name__ == "__main__":
    main()
