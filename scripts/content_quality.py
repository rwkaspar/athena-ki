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


def link_word_ratio(html: str) -> float:
    """Anteil der Wörter, die innerhalb von <a>…</a> stehen, an allen Wörtern.
    Das verlässlichste Signal Index-Seite vs. Dokument: Navigations-/Index-
    Seiten sind fast nur Links (~0.8), echte Dokumente kaum (~0.1).
    Liefert -1, wenn kein HTML übergeben (Signal nicht verfügbar)."""
    if not html:
        return -1.0
    link_text = " ".join(re.findall(r"<a\b[^>]*>(.*?)</a>", html, re.S | re.I))
    link_text = re.sub(r"<[^>]+>", " ", link_text)
    lw = len(link_text.split())
    plain = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    plain = re.sub(r"<[^>]+>", " ", plain)
    total = len(plain.split())
    return lw / total if total else 1.0


def assess(text: str, title: str = "", html: str = "") -> dict:
    """Bewertet einen Seiteninhalt. Liefert dict mit:
      - is_document: bool — True wenn echtes Dokument (genug Fließtext)
      - is_error: bool — Fehler-/Schutzseite
      - reason: str — kurze Begründung
      - metrics: rohe Kennzahlen

    Wenn html übergeben wird, ist die Link-Wort-Dichte das Hauptkriterium
    (zuverlässiger als die Text-Heuristik). Ohne html → Text-Heuristik."""
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
    lwr = link_word_ratio(html)  # -1 wenn kein html

    metrics = {
        "len": len(t), "lines": n_lines, "short_line_ratio": round(short_ratio, 2),
        "sentences": sentences, "words": words,
        "avg_words_per_line": round(avg_words_per_line, 1), "boilerplate_hits": boiler_hits,
        "link_word_ratio": round(lwr, 2),
    }

    # 4a) Hauptkriterium wenn HTML da: Link-Wort-Dichte. >0.55 = überwiegend
    #     Links = Navigations-/Index-Seite. Sehr trennscharf (Doku ~0.1, Index ~0.8).
    if lwr >= 0:
        if lwr > 0.55:
            return {"is_document": False, "is_error": False,
                    "reason": f"Navigations-/Index-Seite (Link-Wort-Anteil {int(lwr*100)}%)",
                    "metrics": metrics}
        return {"is_document": True, "is_error": False,
                "reason": f"Dokument (Link-Wort-Anteil {int(lwr*100)}%, {words} Wörter)",
                "metrics": metrics}

    # 4b) Fallback ohne HTML: Text-Heuristik, konservativ (im Zweifel durchlassen).
    if short_ratio > 0.85 and sentences < 8 and words < 250:
        return {"is_document": False, "is_error": False,
                "reason": f"Navigations-/Index-Seite ({int(short_ratio*100)}% kurze Zeilen, {sentences} Sätze, {words} Wörter)",
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
