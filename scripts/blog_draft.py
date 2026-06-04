#!/usr/bin/env python3
"""Athena — Blog-Entwurf generieren.

Erzeugt aus einem Thema/einer aktuellen Entscheidung einen Markdown-Entwurf
für den EVIDENZ-Blog: Faktenlage, Rechtsrahmen, Optionen mit Trade-offs und
(falls vorhanden) die dokumentierte EVIDENZ-Position. Nutzt die laufende
Athena-Chat-API (inkl. garantiertem EVIDENZ-Positions-Kontext).

Der Entwurf wird mit status:draft gespeichert — ein Mensch redigiert ihn,
prüft die Fakten und setzt status:published. KEINE automatische Veröffentlichung.

Aufruf:
    python scripts/blog_draft.py "Soll das Klimageld eingeführt werden?" \\
        [--slug klimageld] [--provider mistral] [--out-dir ../evidenz-partei/blog]
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import date

API = os.getenv("ATHENA_API", "http://100.105.70.24:8765")
OUT_DIR = os.getenv("BLOG_OUT_DIR",
                    os.path.join(os.path.dirname(__file__), "..", "..", "evidenz-partei", "blog"))


def _slugify(s: str) -> str:
    s = s.lower()
    s = (s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:60] or "beitrag"


def generate(topic: str, provider: str) -> tuple[str, list]:
    """Ruft die Athena-Chat-API mit einem Blog-Brief und liefert (text, quellen)."""
    brief = (
        f"Erstelle eine strukturierte Optionsanalyse zu folgender aktueller "
        f"politischer Frage für einen Blog-Beitrag: '{topic}'.\n\n"
        "Gliedere klar in Markdown: ## Worum es geht, ## Faktenlage (mit "
        "Quellenangaben), ## Rechtsrahmen, ## Die Optionen mit Trade-offs "
        "(2-4 Optionen, je Pro/Contra), und falls eine dokumentierte EVIDENZ-"
        "Position vorliegt ## Die EVIDENZ-Position (als Beschluss referiert, "
        "klar getrennt). Keine eigene Empfehlung. Sachlich, präzise, mit Quellen.\n\n"
        "WICHTIG: Nenne konkrete Zahlen (Statistiken, Beträge, Aktenzeichen) NUR, "
        "wenn sie eindeutig im RAG-Quellenkontext belegt sind. Erfinde KEINE Zahlen. "
        "Wo eine Zahl nötig wäre, aber nicht belegt ist, schreibe '[Zahl prüfen]' "
        "statt zu raten. Halte dich kompakt genug, dass die Antwort vollständig "
        "inklusive der EVIDENZ-Position abgeschlossen wird.\n\n"
        "GUARD: Erfinde KEINE EVIDENZ-Position. Wenn im Kontext keine dokumentierte "
        "EVIDENZ-Position zu dieser Frage vorliegt, schreibe im Abschnitt "
        "## Die EVIDENZ-Position ausschließlich: 'EVIDENZ hat zu dieser Frage noch "
        "keine dokumentierte Position beschlossen.' — und erfinde weder "
        "Programm-Abschnitte noch Beschlüsse noch Aktenzeichen."
    )
    payload = json.dumps({"message": brief, "scope": "bund",
                          "provider": provider, "use_memory": False}).encode()
    req = urllib.request.Request(f"{API}/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    text, sources = "", []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "token":
                text += ev.get("text", "")
            elif ev.get("type") == "sources":
                sources = ev.get("sources", [])
            elif ev.get("type") == "error":
                raise RuntimeError(ev.get("message", "?"))
    return text, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("topic", help="Thema / aktuelle politische Frage")
    p.add_argument("--slug", help="URL-Slug (sonst aus Thema abgeleitet)")
    p.add_argument("--provider", default="mistral", choices=["ollama", "mistral"],
                   help="mistral = faktentreuer (empfohlen für Blog), ollama = lokal")
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    print(f"[blog_draft] generiere Entwurf zu: {args.topic} (provider={args.provider}) …", file=sys.stderr)
    text, sources = generate(args.topic, args.provider)

    today = date.today().isoformat()
    slug = args.slug or _slugify(args.topic)
    fname = f"{today}-{slug}.md"
    title = args.topic.strip()
    # teaser = erste echte Prosa-Zeile (kein ---, keine ##-Überschrift, kein >)
    teaser = ""
    for ln in (l.strip() for l in text.strip().splitlines()):
        if not ln or ln.startswith(("---", "#", ">", "|", "*", "-")):
            continue
        teaser = ln[:200]
        break

    src_md = "\n".join(f"- {s}" for s in sources) if sources else "- (Quellen prüfen)"
    front = (
        "---\n"
        f"title: {title}\n"
        f"date: {today}\n"
        f"topic: {args.topic.strip()}\n"
        "status: draft\n"
        f"teaser: {teaser}\n"
        "---\n\n"
        "> **ENTWURF — vor Veröffentlichung redaktionell prüfen!** KI-generiert "
        "von Athena, Fakten und Quellen gegenchecken, dann status: published setzen.\n\n"
        f"{text.strip()}\n\n## Quellen\n{src_md}\n"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(front)
    print(f"[blog_draft] → {out_path}")
    print(f"[blog_draft] {len(text)} Zeichen, {len(sources)} Quellen. "
          "Redigieren, dann status: published.", file=sys.stderr)


if __name__ == "__main__":
    main()
