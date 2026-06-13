#!/usr/bin/env python3
"""Athena — leichtgewichtige Sprach-Erkennung (ohne externe Dependency).

Zweck: Fremdsprachige Quellen MARKIEREN (nicht filtern). Eine arabische
Bundestags-Übersetzung etwa soll nicht stumm verworfen werden, sondern als
"fremdsprachig — wartet auf sprachkundige Prüfung" in der Queue auftauchen.

Strategie:
- Nicht-lateinische Schrift wird zuverlässig per Unicode-Block erkannt
  (Arabisch, Kyrillisch, CJK, Griechisch, Hebräisch, Devanagari).
- Bei lateinischer Schrift unterscheidet eine Stopwort-/Umlaut-Heuristik
  grob Deutsch vs. Englisch. Genauer wird es bei Bedarf mit langdetect —
  hier bewusst dependency-frei gehalten.

detect(text) -> {"code": "ar", "name": "Arabisch", "confidence": 0.0..1.0, "latin": bool}
"""
import re

_SCRIPTS = [
    ("ar", "Arabisch",   re.compile(r"[؀-ۿݐ-ݿ]")),
    ("ru", "Kyrillisch", re.compile(r"[Ѐ-ӿ]")),
    ("zh", "CJK",        re.compile(r"[一-鿿぀-ヿ가-힯]")),
    ("el", "Griechisch", re.compile(r"[Ͱ-Ͽ]")),
    ("he", "Hebräisch",  re.compile(r"[֐-׿]")),
    ("hi", "Devanagari", re.compile(r"[ऀ-ॿ]")),
]

_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")

_DE = set("der die das und nicht von zu mit für ist auf dem den ein eine werden "
          "des im am sich auch oder als wird sind durch nach bei einer".split())
_EN = set("the and of to in is for that with are this as be by an it from at on "
          "which or have has not".split())


def _words(text):
    return re.findall(r"[a-zà-ÿäöüß]+", text.lower())


def detect(text: str) -> dict:
    text = text or ""
    if len(text.strip()) < 12:
        return {"code": "unknown", "name": "unbekannt", "confidence": 0.0, "latin": True}

    alpha = _LATIN.findall(text)
    n_latin = len(alpha)

    # 1) Nicht-lateinische Schrift? (dominanter Block gewinnt)
    best = None
    for code, name, rx in _SCRIPTS:
        hits = len(rx.findall(text))
        if hits and (best is None or hits > best[2]):
            best = (code, name, hits)
    if best:
        code, name, hits = best
        non_latin_share = hits / max(hits + n_latin, 1)
        if non_latin_share >= 0.30:
            return {"code": code, "name": name,
                    "confidence": round(min(1.0, non_latin_share), 2), "latin": False}

    # 2) Lateinische Schrift → Deutsch vs. Englisch (Stopwort-Heuristik)
    ws = _words(text)
    if not ws:
        return {"code": "unknown", "name": "unbekannt", "confidence": 0.0, "latin": True}
    de = sum(1 for w in ws if w in _DE)
    en = sum(1 for w in ws if w in _EN)
    umlaut = len(re.findall(r"[äöüß]", text)) > 2  # starkes Deutsch-Signal
    if umlaut:
        de += 3
    total = max(de + en, 1)
    if de == 0 and en == 0:
        return {"code": "unknown", "name": "unbekannt", "confidence": 0.2, "latin": True}
    if de >= en:
        return {"code": "de", "name": "Deutsch", "confidence": round(de / total, 2), "latin": True}
    return {"code": "en", "name": "Englisch", "confidence": round(en / total, 2), "latin": True}


# Sprachen, die der RAG (deutsch) ohne sprachkundige Prüfung NICHT braucht.
def is_foreign(code: str) -> bool:
    return code not in ("de", "unknown")


if __name__ == "__main__":
    import sys
    print(detect(sys.stdin.read()))
