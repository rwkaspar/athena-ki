#!/usr/bin/env python3
"""Batch-Treiber: Wirkungs- & Umsetzungsanalyse für ALLE 34 Maßnahmen des
Koalitionsausschuss-Papers „Ein Programm für Aufschwung und Beschäftigung" (2.7.2026).

Lädt Modelle + Vektorstore EINMAL und schickt jede Maßnahme durch analyze()
(3-Familien: Mistral-Erzeugung, gemma4-Adversarial, athena-Critique, Konsolidierung).
Schreibt je Maßnahme eine JSON-Datei; ist RESUMEBAR (fertige Maßnahmen werden
übersprungen), damit ein Abbruch nicht alles wiederholt.

Aufruf (aitest/Ollama muss laufen):
  OLLAMA_HOST=http://100.101.225.56:11434 MISTRAL_API_KEY=… \
    python scripts/batch_reform_analyse.py [--only 11] [--force]
"""
import argparse, datetime, json, os, sys, time, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = pathlib.Path(__file__).resolve().parent.parent
MASS = ROOT / "eval" / "koalition_massnahmen_2026-07-02.json"
OUTDIR = ROOT / "eval" / "reform_analysen"
PAPER = 'Ein Programm für Aufschwung und Beschäftigung'
PAPER_DATE = "2026-07-02"
# Realwelt-Kontext für den Critique: verhindert Fehlalarme, weil athenas Trainingsstand
# vor dem Paper liegt und es sonst die reale Koalition/das Datum für „fiktiv" hält.
WELTKONTEXT = (
    "REALER HINTERGRUND (kein Trainingswissen nötig): Die geprüfte Maßnahme stammt aus dem "
    f"offiziellen Beschlusspapier „{PAPER}“ des Koalitionsausschusses von CDU/CSU und SPD vom "
    f"{PAPER_DATE}, veröffentlicht vom Bundesfinanzministerium. Diese Regierungskoalition, das "
    "Dokument und das Datum sind REAL — auch wenn sie nach deinem Trainingsstand liegen. "
    "Werte die Existenz des Papers, die Koalition oder das Datum NICHT als Erfindung."
)


def build_vorhaben(m):
    """Formuliert die Maßnahme als prüfbares VORHABEN mit Kontext (Paper, Bereich)."""
    return (f"Maßnahme {m['nr']} von 34 des Koalitionsausschuss-Papers „{PAPER}“ "
            f"(CDU/CSU und SPD, {PAPER_DATE}), Bereich „{m['section']}“:\n\n{m['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="nur diese Maßnahmennummer(n), kommasepariert (z. B. 1,11,28)")
    ap.add_argument("--force", action="store_true", help="auch fertige neu rechnen")
    a = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    massnahmen = json.loads(MASS.read_text(encoding="utf-8"))
    if a.only:
        keep = {int(x) for x in str(a.only).split(",") if x.strip()}
        massnahmen = [m for m in massnahmen if m["nr"] in keep]

    import serve
    from gen_position_umsetzbarkeit import analyze, _make_llms
    print("… lade Vektorstore (bund) + Modelle …", file=sys.stderr, flush=True)
    vs, _ = serve._get_components("bund")
    gen, adv, critique, host = _make_llms()

    total = len(massnahmen)
    done = skipped = failed = 0
    t0 = time.time()
    for i, m in enumerate(massnahmen, 1):
        out = OUTDIR / f"massnahme_{m['nr']:02d}.json"
        if out.exists() and not a.force:
            skipped += 1
            print(f"[{i}/{total}] M{m['nr']:02d} übersprungen (existiert)", file=sys.stderr, flush=True)
            continue
        vorhaben = build_vorhaben(m)
        query = m["text"][:300]
        tm = time.time()
        print(f"[{i}/{total}] M{m['nr']:02d} ({m['section']}) … ", file=sys.stderr, flush=True)
        try:
            rec, _ = analyze(vorhaben, query, vs, gen, adv, critique, host, serve,
                             weltkontext=WELTKONTEXT, heute=datetime.date.today().isoformat())
        except Exception as e:
            failed += 1
            print(f"   [FAIL] M{m['nr']:02d}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        rec_full = {"nr": m["nr"], "section": m["section"], "titel": m["text"][:120],
                    "massnahme_text": m["text"], **rec}
        out.write_text(json.dumps(rec_full, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        dt = time.time() - tm
        rev = " ‼REVISION" if rec.get("revision_needed") else ""
        nfab = len(rec.get("fabrikate", []))
        fab = f" ⚠{nfab}×unbelegt" if nfab else (" ⚠FLAG" if rec.get("fabrikationsverdacht") else "")
        ziel = rec.get("zielerreichung", {}).get("status", "?")
        print(f"   [ok] M{m['nr']:02d}: Ziel={ziel} · Umsetzung={rec['gesamt']}{rev}{fab} · "
              f"{len(rec.get('adversarial', []))} Adv · {len(rec.get('konsolidierung', []))} Konsol · {dt:.0f}s",
              file=sys.stderr, flush=True)

    mins = (time.time() - t0) / 60
    print(f"\n[done] {done} gerechnet, {skipped} übersprungen, {failed} fehlgeschlagen "
          f"in {mins:.1f} min → {OUTDIR}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
