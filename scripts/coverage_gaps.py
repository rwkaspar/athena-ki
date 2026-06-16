#!/usr/bin/env python3
"""Athena — Abdeckungs-Lücken-Sammler (kontinuierliche Verbesserung).

Liest Pipeline-Durchläufe (docs/pipeline_demo_*.json) und sammelt Aussagen, die
NOCH NICHT in der Wissensbasis geerdet sind — also Kandidaten für neue Quellen.
Eine Aussage gilt als „Lücke", wenn die Faktentreue-Prüfung sie nicht bestätigt
(verification_status ∉ {verifiziert}) ODER die Adversarial-Prüfung sie flaggt
(verdict ∉ {haelt}).

WICHTIG (Anti-Konvergenz): Die hier gesammelten Lücken sind ein BACKLOG für
MENSCHLICHE Quellen-Recherche — keine Auto-Ingestion. Mistral-Vorschläge (falls
vorhanden) sind nur ein Hinweis, kein Beleg. Priorität haben Primärquellen, die
Mistral NICHT kennt: aktuelle/post-cutoff, amtlich, nischig.

Output: eval/coverage_gaps.json (maschinell) + eval/coverage_gaps.md (lesbar) +
eine Abdeckungs-Quote je Lauf (steigt sie über Zeit, wird das RAG messbar besser).

Aufruf:  python scripts/coverage_gaps.py
"""
import glob
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO_GLOB = str(ROOT / "docs" / "pipeline_demo_*.json")
OUT_JSON = ROOT / "eval" / "coverage_gaps.json"
OUT_MD = ROOT / "eval" / "coverage_gaps.md"

GROUNDED_VERIFY = {"verifiziert"}
GROUNDED_ADVERSARIAL = {"haelt"}


def _is_grounded(fakt: dict, adv_by_aussage: dict) -> bool:
    vs = (fakt.get("verification_status") or "").strip().lower()
    verify_ok = vs in GROUNDED_VERIFY or bool(fakt.get("verifiziert"))
    adv = adv_by_aussage.get((fakt.get("aussage") or "").strip())
    adv_ok = (adv is None) or ((adv.get("verdict") or "").lower() in GROUNDED_ADVERSARIAL)
    return verify_ok and adv_ok


def collect():
    runs, gaps = [], []
    for path in sorted(glob.glob(DEMO_GLOB)):
        d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        topic = d.get("question", pathlib.Path(path).stem)
        facts = (d.get("analysis") or {}).get("faktenlage") or []
        adv_by = {(a.get("aussage") or "").strip(): a for a in (d.get("adversarial") or [])}
        n = len(facts)
        grounded = sum(1 for f in facts if _is_grounded(f, adv_by))
        runs.append({"datei": os.path.basename(path), "thema": topic,
                     "aussagen": n, "geerdet": grounded,
                     "quote": round(grounded / n, 2) if n else None})
        for f in facts:
            if not _is_grounded(f, adv_by):
                adv = adv_by.get((f.get("aussage") or "").strip()) or {}
                gaps.append({
                    "thema": topic,
                    "aussage": f.get("aussage", ""),
                    "verification_status": f.get("verification_status"),
                    "adversarial_verdict": adv.get("verdict"),
                    "hinweis_quelle": adv.get("quelle") or "",  # nur Hinweis, NICHT Beleg
                    "prioritaet": "primärquelle suchen (amtlich/aktuell bevorzugt)",
                })
    return runs, gaps


def main():
    runs, gaps = collect()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"runs": runs, "gaps": gaps}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    lines = ["# Abdeckungs-Lücken — Quellen-Backlog", "",
             "Automatisch aus Pipeline-Läufen erzeugt. **Menschlich** recherchieren + über die "
             "Verify-Queue ingesten — keine Auto-Übernahme. Priorität: Primärquellen, die Mistral "
             "nicht kennt (aktuell/amtlich/nischig).", "",
             "## Abdeckungs-Quote je Lauf (steigt = RAG wird besser)", ""]
    for r in runs:
        lines.append(f"- **{r['quote']}** — {r['geerdet']}/{r['aussagen']} geerdet · {r['thema'][:70]}")
    lines += ["", f"## Offene Lücken ({len(gaps)})", ""]
    for g in gaps:
        why = g.get("adversarial_verdict") or g.get("verification_status") or "ungeprüft"
        hint = f" · Hinweis: {g['hinweis_quelle']}" if g.get("hinweis_quelle") else ""
        lines.append(f"- [{why}] {g['aussage'][:110]}{hint}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {len(runs)} Läufe, {len(gaps)} Lücken → {OUT_JSON.name} + {OUT_MD.name}")
    for r in runs:
        print(f"   Abdeckung {r['quote']}  ({r['geerdet']}/{r['aussagen']})  {r['thema'][:55]}")


if __name__ == "__main__":
    main()
