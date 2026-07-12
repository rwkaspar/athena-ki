#!/usr/bin/env python3
"""Options-Analyse „neues Steuersystem" — Mistral (Athena) + RAG.

Zweiteilig: (A) Diagnose des aktuellen deutschen Steuersystems (Schwächen +
Potenziale), (B) potenzielle neue Steuersysteme mit Stärken/Schwächen, bewertet
an drei Kriterien: FAIRNESS, SCHLUPFLÖCHER, WETTBEWERBSFÄHIGKEIT.
EVIDENZ-neutral: keine Empfehlung, Optionen mit Trade-offs + Wertannahmen.

Aufruf:
  OLLAMA_HOST=… MISTRAL_API_KEY=… python scripts/analyse_steuersystem.py --out eval/steuersystem_analyse.md
"""
import argparse, datetime, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

QUERIES = [
    "deutsches Steuersystem Struktur Einkommensteuer Progression kalte Progression Mittelstandsbauch",
    "Steuerschlupflöcher Steuervermeidung Gestaltung Share Deals Grunderwerbsteuer Erbschaftsteuer Verschonung Betriebsvermögen",
    "Abgeltungsteuer Kapitalerträge vs Arbeitseinkommen Besteuerung Ungleichheit Vermögen",
    "Unternehmensbesteuerung Wettbewerbsfähigkeit Steuerwettbewerb Gewinnverlagerung international Mindeststeuer",
    "Steuerreform Modelle Flat Tax duale Einkommensteuer Konsumsteuer Vereinfachung Grundfreibetrag",
    "Abgabenlast Arbeit Sozialabgaben Steuerquote Deutschland Vergleich OECD",
]

PROMPT = """Heute ist der {heute}. Du bist die Analyse-KI Athena einer faktenbasierten, parteiunabhängigen \
Bewegung (EVIDENZ). Erarbeite eine strukturierte OPTIONS-ANALYSE zu einem möglichen neuen Steuersystem für \
Deutschland. Du BEWERTEST und STRUKTURIERST, sprichst aber KEINE Empfehlung aus — lege stattdessen die \
Wertannahmen offen, an denen die Entscheidung hängt.

QUELLENDISZIPLIN (wichtigste Regel): Konkrete empirische Angaben — Steuersätze, Aufkommenszahlen, Quoten, \
€-Beträge, Studienergebnisse — nur nennen, wenn sie in den QUELLEN stehen oder gesichertes Lehrbuchwissen \
sind. Erfinde KEINE Zahlen. Wo eine Zahl die Aussage trägt, aber unbelegt ist, schreibe „(Größenordnung zu \
belegen)". Qualitatives ökonomisches Schließen ist erlaubt; erfundene Statistiken nicht.

FOKUS-KRITERIEN (an jedem Modell anlegen): (1) FAIRNESS — vertikal (Leistungsfähigkeit) und horizontal \
(gleiche Einkommen gleich besteuert); (2) SCHLUPFLÖCHER — wie gestaltungs-/vermeidungsanfällig; (3) \
WETTBEWERBSFÄHIGKEIT — Standort, Kapital-/Arbeitsanreize, internationale Mobilität. Zusätzlich kurz: \
Praktikabilität/Verwaltbarkeit.

════ TEIL A — DIAGNOSE DES AKTUELLEN SYSTEMS ════
1. Kurzer Überblick der tragenden Steuerarten (Einkommensteuer, Körperschaft-/Gewerbesteuer, Umsatzsteuer, \
   Abgeltungsteuer, Erbschaft-/Schenkungsteuer, Grunderwerbsteuer, Sozialabgaben als Kontext).
2. SCHWÄCHEN — die gravierendsten, je 1–2 Sätze mit Begründung (z. B. Kapital vs. Arbeit ungleich belastet, \
   kalte Progression/Mittelstandsbauch, Schlupflöcher wie Share Deals oder Betriebsvermögens-Verschonung, \
   hohe Abgabenlast auf Arbeit, Komplexität, Aufkommensverteilung).
3. POTENZIALE — wo ließe sich fairer/schlupflochärmer/wettbewerbsfähiger ansetzen (Basis verbreitern, \
   Lücken schließen, Lastverschiebung Arbeit→Vermögen/Konsum/Umwelt, Vereinfachung).

════ TEIL B — POTENZIELLE NEUE STEUERSYSTEME ════
Stelle 5–7 grundsätzlich verschiedene Modelle dar (z. B. Flat Tax mit hohem Grundfreibetrag; Duale \
Einkommensteuer nach nordischem Vorbild; synthetische Einkommensteuer mit voller Gleichbehandlung aller \
Einkunftsarten; konsum-/cash-flow-orientierte Besteuerung; ökologisch-soziale Steuerreform mit \
Entlastung der Arbeit; vermögens-/erbschaftsteuer-fokussierte Reform; destination-based cash-flow tax \
für Unternehmen; negative Einkommensteuer). Für JEDES Modell:
- **Kurzbeschreibung** (2–3 Sätze: wie funktioniert es).
- **Fairness**, **Schlupflöcher**, **Wettbewerbsfähigkeit** — je eine ehrliche Einschätzung (auch Nachteile).
- **Stärken** und **Schwächen** (Stichpunkte).
- **Wertannahme**: welche politische Grundüberzeugung müsste man teilen, um dieses Modell zu wählen.

════ ABSCHLUSS ════
- Ein kurzer QUERSCHNITT: Welche Trade-offs kehren wieder (z. B. Einfachheit vs. Zielgenauigkeit; \
  Umverteilung vs. Kapitalmobilität; Aufkommen vs. Anreize)?
- KEINE Empfehlung — benenne die zentralen Wertfragen, an denen die Wahl hängt.

Antworte als sauberes MARKDOWN (Überschriften ##/###, Tabellen wo sinnvoll), auf Deutsch.

QUELLEN (RAG):
{context}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/steuersystem_analyse.md")
    a = ap.parse_args()

    import serve
    from retrieval import tier_aware_retrieve, format_docs
    from langchain_mistralai import ChatMistralAI
    vs, _ = serve._get_components("bund")
    docs, seen = [], set()
    for q in QUERIES:
        for d in tier_aware_retrieve(vs, q, k=serve.RETRIEVER_K, fetch_k=50, sim_floor=0.35, max_k=8):
            key = (d.metadata.get("source", ""), d.page_content[:80])
            if key not in seen:
                seen.add(key); docs.append(d)
    ctx = format_docs(docs)
    print(f"… {len(docs)} Kontext-Chunks aus {len(set(d.metadata.get('source') for d in docs))} Quellen", file=sys.stderr)

    gen = ChatMistralAI(model="mistral-large-latest", api_key=os.environ["MISTRAL_API_KEY"],
                        temperature=0.1, max_tokens=6000, timeout=300)
    resp = gen.invoke(PROMPT.replace("{heute}", datetime.date.today().isoformat()).replace("{context}", ctx[:14000]))
    md = getattr(resp, "content", resp)
    quellen = sorted(set(d.metadata.get("source", "") for d in docs if d.metadata.get("source")))
    md += "\n\n## Verwendete Quellen (RAG)\n" + "\n".join(f"- {q}" for q in quellen)
    open(a.out, "w", encoding="utf-8").write(md)
    print(f"[ok] → {a.out} ({len(md)} Zeichen)", file=sys.stderr)


if __name__ == "__main__":
    main()
