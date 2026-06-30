#!/usr/bin/env python3
"""Für die in der Gegenprüfung als UNBELEGT/zweifelhaft markierten Faktenaussagen bei
Mistral nachhaken: „Worauf stützt sich diese Zahl?" → konkrete Quellen-VORSCHLÄGE.

WICHTIG: Mistral kann Quellen halluzinieren. Die Ausgabe ist ein RECHERCHE-Startpunkt,
KEIN Beleg. Mit --submit werden die vorgeschlagenen URLs als Submissions angelegt und
durchlaufen den normalen Verify-Weg (auto_review → menschliche Freigabe → ingest) —
erst dann gilt eine Aussage als belegt.

Aufruf:  OLLAMA_HOST=… MISTRAL_API_KEY=… python scripts/gen_beleg_nachfrage.py [--only slug] [--submit]
"""
import argparse, json, os, re, sys, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT = ROOT / "eval" / "position_gegenpruefung.jsonl"
OUT = ROOT / "eval" / "belegluecken.jsonl"

PROMPT = ("In einer politischen Analyse wurde diese konkrete Aussage gemacht:\n«{aussage}»\n\n"
    "Worauf stützt sich diese Aussage faktisch? Nenne die konkretesten ÜBERPRÜFBAREN "
    "Primärquellen (Herausgeber/Institution, Titel, Jahr, und — falls du sie sicher kennst — "
    "URL oder DOI). Erfinde NICHTS: Wenn du keine belastbare Quelle sicher kennst, setze "
    "\"unsicher\": true und nenne höchstens, WO man typischerweise suchen würde (Institution/"
    "Statistik), aber KEINE erfundene URL.\n"
    "Antworte NUR als JSON: {\"quellen\":[{\"herausgeber\":\"\",\"titel\":\"\",\"jahr\":\"\",\"url\":\"\"}],"
    "\"unsicher\":true|false,\"hinweis\":\"<1 Satz, wo zu suchen>\"}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="nur dieser slug")
    ap.add_argument("--submit", action="store_true", help="vorgeschlagene URLs als Submissions anlegen (Verify-Queue)")
    a = ap.parse_args()
    if not REPORT.exists():
        print("Kein Gegenprüf-Report — erst gen_position_gegenpruefung.py laufen lassen.", file=sys.stderr)
        return

    from langchain_mistralai import ChatMistralAI
    llm = ChatMistralAI(model="mistral-large-latest", api_key=os.environ["MISTRAL_API_KEY"],
                        temperature=0.0, max_tokens=900, timeout=120)

    # je slug die jüngste Report-Zeile
    latest = {}
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line); latest[r["slug"]] = r
            except Exception:
                pass

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_open = n_submit = 0
    for slug, r in latest.items():
        if a.only and slug != a.only:
            continue
        offen = [v for v in r.get("verdicts", []) if v.get("verdict") != "haelt"]
        if not offen:
            continue
        print(f"… {slug}: {len(offen)} offene Aussage(n)", file=sys.stderr)
        for v in offen:
            aussage = v.get("aussage", "")
            try:
                raw = llm.invoke(PROMPT.replace("{aussage}", aussage)).content
                d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
            except Exception as e:
                print(f"  [fail] {aussage[:50]}: {type(e).__name__}", file=sys.stderr); continue
            rec = {"slug": slug, "aussage": aussage, "verdict": v.get("verdict"),
                   "quellen_vorschlag": d.get("quellen", []), "unsicher": bool(d.get("unsicher")),
                   "hinweis": d.get("hinweis", ""), "status": "ungeprueft"}
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_open += 1
            mark = "❓ unsicher" if rec["unsicher"] else f"{len(rec['quellen_vorschlag'])} Quelle(n)"
            print(f"  • {aussage[:60]} → {mark}", file=sys.stderr)
            if a.submit:
                try:
                    from crawl_ingest import _make_submission
                    for q in rec["quellen_vorschlag"]:
                        url = (q.get("url") or "").strip()
                        if url.startswith("http"):
                            _make_submission(url, "bund", f"beleg-nachfrage:{slug}")
                            n_submit += 1
                except Exception as e:
                    print(f"  [submit-fail] {type(e).__name__}: {e}", file=sys.stderr)
    print(f"[done] {n_open} offene Aussagen abgefragt → {OUT}"
          + (f" · {n_submit} URL-Submissions angelegt (Verify-Queue)" if a.submit else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
