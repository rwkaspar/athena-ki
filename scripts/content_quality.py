#!/usr/bin/env python3
"""Athena — Inhaltsqualitäts-Heuristik: echtes Dokument vs. Navigations-/Index-Seite.

Wird gebraucht von:
- crawl.py (welche gecrawlten Seiten sind echte Dokumente?)
- ingest.py / auto_review.py (Ingest-Schutz: keine Navigations-Seiten aufnehmen)

Kein LLM — reine, schnelle Heuristik über Textstruktur. Idee: Navigations-/
Index-Seiten bestehen überwiegend aus kurzen Link-Texten und Menü-Phrasen,
echte Dokumente aus längerem Fließtext mit Sätzen.
"""

import re

# Phrasen, die typisch für Navigation/Boilerplate sind (deutsch + engl.)
BOILERPLATE_PHRASES = [
    "springe direkt zu", "zur navigation", "zum inhalt", "hauptmenü", "servicemenü",
    "cookie", "datenschutz", "impressum", "barrierefreiheit", "leichte sprache",
    "newsletter abonnieren", "zur startseite", "skip to content", "main menu",
    "alle publikationen", "alle themen", "übersicht:", "weitere publikationen",
    "mehr erfahren", "mehr lesen", "weiterlesen", "zurück zur übersicht",
]

# Wörter, die auf Fehler-/Schutzseiten hindeuten
ERROR_MARKERS = [
    "404", "not found", "seite nicht gefunden", "page not found", "http status 404",
    "error occurred", "internal server error", "security checkpoint", "forbidden",
    "access denied", "zugriff verweigert", "503", "service unavailable",
]


def _sentences(text: str) -> int:
    """Grobe Satz-Zählung: Folgen, die mit Satzzeichen enden und >30 Zeichen haben."""
    parts = re.split(r"[.!?]\s", text)
    return sum(1 for p in parts if len(p.strip()) > 40)


def assess(text: str, title: str = "") -> dict:
    """Bewertet einen Seiteninhalt. Liefert dict mit:
      - is_document: bool — True wenn echtes Dokument (genug Fließtext)
      - is_error: bool — Fehler-/Schutzseite
      - reason: str — kurze Begründung
      - metrics: rohe Kennzahlen
    """
    t = (text or "").strip()
    low = t.lower()
    title_low = (title or "").lower()

    # 1) Fehlerseite?
    if any(mk in title_low for mk in ERROR_MARKERS) or (len(t) < 400 and any(mk in low for mk in ERROR_MARKERS)):
        return {"is_document": False, "is_error": True,
                "reason": "Fehler-/Schutzseite (Titel/Inhalt enthält Fehlermarker)",
                "metrics": {"len": len(t)}}

    # 2) zu kurz für ein Dokument
    if len(t) < 600:
        return {"is_document": False, "is_error": False,
                "reason": f"zu wenig Text ({len(t)} Zeichen)",
                "metrics": {"len": len(t)}}

    # 3) Struktur-Metriken
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    n_lines = len(lines)
    short_lines = sum(1 for l in lines if len(l) < 40)          # kurze Zeilen = Menüpunkte
    short_ratio = short_lines / n_lines if n_lines else 1.0
    sentences = _sentences(t)
    words = len(t.split())
    avg_words_per_line = words / n_lines if n_lines else 0
    boiler_hits = sum(1 for p in BOILERPLATE_PHRASES if p in low)

    metrics = {
        "len": len(t), "lines": n_lines, "short_line_ratio": round(short_ratio, 2),
        "sentences": sentences, "words": words,
        "avg_words_per_line": round(avg_words_per_line, 1), "boilerplate_hits": boiler_hits,
    }

    # 4) Entscheidung: Navigations-Seite, wenn überwiegend kurze Zeilen + wenig Sätze
    #    Echtes Dokument: viele vollständige Sätze, längere Zeilen.
    if short_ratio > 0.75 and sentences < 15:
        return {"is_document": False, "is_error": False,
                "reason": f"Navigations-/Index-Seite ({int(short_ratio*100)}% kurze Zeilen, nur {sentences} Sätze)",
                "metrics": metrics}
    if sentences < 8 and avg_words_per_line < 6:
        return {"is_document": False, "is_error": False,
                "reason": f"kaum Fließtext ({sentences} Sätze, {avg_words_per_line} Wörter/Zeile)",
                "metrics": metrics}

    return {"is_document": True, "is_error": False,
            "reason": f"Dokument ({sentences} Sätze, {words} Wörter)",
            "metrics": metrics}


if __name__ == "__main__":
    import sys, json
    # Selbsttest / manuelle Prüfung: Text von stdin oder URL-Argument
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        import requests
        html = requests.get(sys.argv[1], timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
        text = re.sub(r"<[^>]+>", "\n", text)
        import html as H
        text = H.unescape(text)
    else:
        text = sys.stdin.read()
    print(json.dumps(assess(text), ensure_ascii=False, indent=2))
