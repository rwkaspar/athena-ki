#!/usr/bin/env python3
"""Erzeugt Wert-Wirkungs-VORSCHLÄGE (§§1–7, -100..+100) je EVIDENZ-Programmposition
für die Säulendiagramme auf programm.html. Pro Thema EIN LLM-Call: Positionstext +
Wertekanon → JSON. Inkrementell gespeichert (Abbruch-sicher). Ergebnis ist ein
ENTWURF zur menschlichen Prüfung — keine erfundenen Zahlen ohne Freigabe.

Aufruf:  OLLAMA_HOST=… python scripts/gen_position_wertwirkung.py [--only slug] [--model athena:latest]
"""
import argparse
import json
import os
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_MD = ROOT / "output"
WW_JSON = ROOT.parent / "evidenz-partei" / "data" / "wertwirkung.json"
CANON = (ROOT / "config" / "wertekanon.md").read_text(encoding="utf-8")

POSITIONS = [
    ("parteiprogramm_01_direktdemokratie.md", "direkte-demokratie"),
    ("parteiprogramm_02_wahlrecht.md", "wahlrecht"),
    ("parteiprogramm_03_parteienfinanzierung.md", "parteienfinanzierung"),
    ("parteiprogramm_04_schuldenbremse.md", "schuldenbremse"),
    ("parteiprogramm_05_steuersystem.md", "steuersystem"),
    ("parteiprogramm_05a_ki_regulation.md", "ki-regulation"),
    ("parteiprogramm_06_buergergeld.md", "buergergeld"),
    ("parteiprogramm_07_parlamentarismus.md", "parlamentarismus"),
    ("parteiprogramm_07a_abgeordnete_diaeten.md", "abgeordnete-diaeten"),
    ("parteiprogramm_08_rente.md", "rente"),
    ("parteiprogramm_09_klima_energie.md", "klima-energie"),
    ("parteiprogramm_10_migration_asyl.md", "migration-asyl"),
    ("parteiprogramm_11_verteidigung.md", "verteidigung"),
    ("parteiprogramm_12_gesundheit_buergerversicherung.md", "gesundheit"),
]

PROMPT = """Du bewertest, wie EINE politische Position der Partei EVIDENZ auf die sieben \
Paragraphen ihres Wertekanons wirkt. Das ist KEINE Empfehlung und kein Urteil von außen — \
es ist die transparente Selbsteinordnung: welche eigenen Grundwerte diese Position stützt \
oder belastet.

Gib für JEDEN der sieben Paragraphen einen Wert von -100 bis +100:
+100 = die Position stützt diesen Wert stark, 0 = neutral/nicht berührt, -100 = sie belastet \
ihn stark. Sei ehrlich auch bei Belastungen — jede Position hat einen Preis.

Antworte NUR als JSON:
{"wertwirkung":[{"paragraph":"§1","intensitaet":<int>,"begruendung":"<1 kurzer Satz>"}, … alle §1–§7]}

WERTEKANON (§§1–7):
{canon}

POSITION (EVIDENZ-Programm):
{position}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="nur dieser slug")
    ap.add_argument("--model", default=os.getenv("WW_MODEL", "athena:latest"))
    a = ap.parse_args()

    from langchain_ollama import OllamaLLM
    host = os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434")
    llm = OllamaLLM(model=a.model, base_url=host, timeout=600,
                    num_ctx=16384, num_predict=900, reasoning=False, format="json")

    data = {}
    if WW_JSON.exists():
        try:
            data = json.loads(WW_JSON.read_text(encoding="utf-8"))
        except ValueError:
            data = {}

    valid = {f"§{i}" for i in range(1, 8)}
    for fname, slug in POSITIONS:
        if a.only and slug != a.only:
            continue
        md = OUT_MD / fname
        if not md.exists():
            print(f"[skip] {slug}: {fname} fehlt", file=sys.stderr)
            continue
        pos = md.read_text(encoding="utf-8")[:8000]
        print(f"… {slug}", file=sys.stderr)
        try:
            raw = llm.invoke(PROMPT.replace("{canon}", CANON).replace("{position}", pos))
            d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            ww = []
            for w in d.get("wertwirkung", []):
                p = (w.get("paragraph") or "").strip()
                if p in valid:
                    v = max(-100, min(100, int(round(float(w.get("intensitaet", 0))))))
                    ww.append({"paragraph": p, "intensitaet": v,
                               "begruendung": (w.get("begruendung") or "").strip()[:200]})
            if len(ww) < 4:
                print(f"  [warn] {slug}: nur {len(ww)} Werte — übersprungen", file=sys.stderr)
                continue
            data[slug] = sorted(ww, key=lambda x: x["paragraph"])
            WW_JSON.parent.mkdir(parents=True, exist_ok=True)
            WW_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  [ok] {slug}: {len(ww)} Werte gespeichert", file=sys.stderr)
        except Exception as e:
            print(f"  [fail] {slug}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"[done] {len(data)} Themen in {WW_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
