#!/usr/bin/env python3
"""Holt die adversariale Gegenprüfung für Programm-Positionen nach, die noch keine
haben. Pro Position: zentrale Faktenaussagen extrahieren → mit einem UNABHÄNGIGEN
Modell (athena, andere Familie als das erstellende Mistral) gegen die RAG-Quellen
adversarial prüfen (haelt = Quellen widersprechen NICHT; widerlegt = Quellen
widersprechen). Aktualisiert die Status-Zeile faktengetreu und schreibt einen
Prüf-Report nach eval/position_gegenpruefung.jsonl.

Aufruf:  OLLAMA_HOST=… python scripts/gen_position_gegenpruefung.py [--only slug] [--limit N]
"""
import argparse, json, os, re, sys, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_MD = ROOT / "output"
REPORT = ROOT / "eval" / "position_gegenpruefung.jsonl"

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

EXTRACT = ("Lies die folgende politische Position. Extrahiere die 5–8 zentralen "
    "ÜBERPRÜFBAREN IST-FAKTENAUSSAGEN über die WELT — also Aussagen über den aktuellen "
    "Zustand, die Rechtslage oder empirische/prognostizierte Größen, die man gegen Quellen "
    "prüfen kann. BEISPIELE für extrahieren: 'Das heutige Rentenniveau beträgt 48 %', "
    "'Das DIW prognostiziert 20 % Altersarmut bis 2040'. "
    "NICHT extrahieren: was die Partei FORDERT, WILL, PLANT oder BESCHLIESST "
    "('… wird auf 53 % angehoben', '… soll eingeführt werden', '… bleibt bei 67') — das sind "
    "Forderungen/Absichten, KEINE überprüfbaren Ist-Aussagen. Faustregel: enthält die Aussage "
    "ein partei-seitiges 'soll/wird/wollen/beschließt/bleibt' → NICHT extrahieren. "
    "Antworte NUR als JSON: {\"aussagen\":[\"<knappe Ist-Faktaussage>\", …]}\n\nPOSITION:\n{md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--limit", type=int, default=99)
    a = ap.parse_args()

    import serve
    from retrieval import tier_aware_retrieve, format_docs
    from pipeline_demo import adversarial_verify, critique_verdict
    from critique import create_critique_chain
    from langchain_ollama import OllamaLLM
    host = os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434")
    vs, _ = serve._get_components("bund")
    extractor = OllamaLLM(model=os.getenv("EXTRACT_MODEL", "athena:latest"), base_url=host,
                          timeout=600, num_ctx=16384, num_predict=900, reasoning=False, format="json")
    # Critique = unabhängiges Modell (athena, andere Familie als das erstellende Mistral)
    critique_fn = create_critique_chain(model=os.getenv("CRITIQUE_MODEL", "athena:latest"), host=host)

    done = 0
    for fname, slug in POSITIONS:
        if a.only and slug != a.only:
            continue
        if done >= a.limit:
            break
        md_path = OUT_MD / fname
        md = md_path.read_text(encoding="utf-8")
        status = next((l for l in md.splitlines() if l.startswith("**Status:")), "")
        if re.search(r"gegengepr|adversarial", status, re.I):
            print(f"[skip] {slug}: schon gegengeprüft", file=sys.stderr); continue
        # Frage als Retrieval-Query
        fm = re.search(r"(?:Entscheidungsfrage|Worum geht's)[:\s*]*\n?(.+)", md)
        query = (fm.group(1) if fm else slug).strip()[:300]
        print(f"… {slug}: extrahiere Aussagen", file=sys.stderr)
        try:
            raw = extractor.invoke(EXTRACT.replace("{md}", md[:9000]))
            aussagen = json.loads(re.search(r"\{.*\}", raw, re.S).group(0)).get("aussagen", [])
        except Exception as e:
            print(f"  [fail-extract] {slug}: {type(e).__name__}", file=sys.stderr); continue
        aussagen = [s for s in aussagen if isinstance(s, str) and s.strip()][:8]
        if not aussagen:
            print(f"  [warn] {slug}: keine Aussagen", file=sys.stderr); continue
        docs = tier_aware_retrieve(vs, query, k=serve.RETRIEVER_K, fetch_k=50, sim_floor=0.4, max_k=12)
        ctx = format_docs(docs)
        # 1) Adversarial: Aussage für Aussage gegen die Quellen
        print(f"  adversarial: {len(aussagen)} Aussagen gegen {len(docs)} Chunks …", file=sys.stderr)
        verdicts = adversarial_verify([{"aussage": s} for s in aussagen], ctx, host)
        halten = sum(1 for v in verdicts if v.get("verdict") == "haelt")
        probleme = [v for v in verdicts if v.get("verdict") != "haelt"]
        n = len(verdicts)
        # 2) Critique: holistischer methodischer Review der ganzen Position
        print(f"  critique: holistischer Review …", file=sys.stderr)
        try:
            tagged = ("[Hinweis: Dies ist eine beschlossene EVIDENZ-PARTEIPOSITION. Sie enthält "
                      "bewusst politische Forderungen und Zielwerte — das sind KEINE Faktenbehauptungen "
                      "und niemals Faktenfehler. Prüfe nur die überprüfbaren Ist-/Faktenbehauptungen.]\n\n"
                      + md[:9000])
            crit = critique_fn(query, docs, tagged)
            cverd = critique_verdict(crit, host)
        except Exception as e:
            print(f"  [warn] critique fehlgeschlagen: {type(e).__name__}", file=sys.stderr)
            crit, cverd = "", {}
        crit_problem = bool(cverd.get("erfundene_fakten"))
        # Status-Zeile aktualisieren (Critique + Adversarial)
        suffix = f"{halten}/{n} Aussagen halten"
        if probleme:
            suffix += f", {len(probleme)} offen"
        if crit_problem:
            suffix += "; Critique: Hinweise"
        new_status = status.rstrip() + f" · Critique + Adversarial gegengeprüft (athena: {suffix})"
        md = md.replace(status, new_status, 1)
        md_path.write_text(md, encoding="utf-8")
        # Report
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT, "a", encoding="utf-8") as f:
            f.write(json.dumps({"slug": slug, "n": n, "halten": halten, "verdicts": verdicts,
                                "critique_verdict": cverd, "critique": crit}, ensure_ascii=False) + "\n")
        flag = "✓" if not probleme and not crit_problem else f"⚠ {len(probleme)} Aussagen offen{' + Critique-Hinweise' if crit_problem else ''}"
        print(f"  [ok] {slug}: {halten}/{n} halten · Critique {'⚠' if crit_problem else 'ok'} {flag}", file=sys.stderr)
        if cverd.get("fazit"):
            print(f"      Critique-Fazit: {cverd['fazit'][:110]}", file=sys.stderr)
        for p in probleme:
            print(f"      ! {p.get('verdict')}: {p.get('aussage','')[:65]} — {p.get('begruendung','')[:75]}", file=sys.stderr)
        done += 1
    print(f"[done] {done} Positionen geprüft", file=sys.stderr)


if __name__ == "__main__":
    main()
