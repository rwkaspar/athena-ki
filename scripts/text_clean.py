#!/usr/bin/env python3
"""Athena — Text-Normalisierung gegen typische PDF/HTML-Encoding-Probleme.

Behebt die Befunde aus der Bestand-Diagnose:
  * Mojibake (UTF-8 als Latin-1 dekodiert: „Ã¤" statt „ä")
  * PUA-Zeichen aus PDF-Schriften ohne Unicode-Mapping (U+E000–U+F8FF)
  * Replacement-Chars „�" (U+FFFD)
  * Typografische Ligaturen (ﬁ ﬂ ﬃ ﬄ ﬀ) statt aufgelöster Bigramme
  * Eingestreute Soft-Hyphens (U+00AD) und Zero-Width-Chars
  * BOMs in der Textmitte

Wird vor dem Chunking aufgerufen (ingest.py) und kann den Bestand
nachträglich bereinigen (audit_encoding.py).
"""
import re
import unicodedata

# Typografische Ligaturen auflösen — Standard-PDF-Problem
_LIG = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st", "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE",
}
_LIG_RE = re.compile("|".join(map(re.escape, _LIG)))

# Zero-Width + Soft-Hyphen + BOM mitten im Text → raus
_INVISIBLES = re.compile(r"[­​‌‍﻿⁠]")

# Replacement-Char (PDF-Extraktor konnte Zeichen nicht decodieren) → raus
_REPLACE_CHAR = re.compile("�+")

# PUA: U+E000–U+F8FF (BMP) + U+F0000–U+FFFFD + U+100000–U+10FFFD
_PUA = re.compile(r"[-\U000f0000-\U000ffffd\U00100000-\U0010fffd]")

# Klassischer UTF-8-als-Latin1-Mojibake: erkennbar an "Ã"-Vorlauf
_MOJIBAKE_HINT = re.compile(r"Ã[-¿]|â[-]|Â[ -¿]")


def _try_ftfy(text: str) -> tuple[str, bool]:
    """Wenn ftfy verfügbar ist, nutzt es das (robuster). Sonst Heuristik."""
    try:
        import ftfy  # optional
        fixed = ftfy.fix_text(text)
        return fixed, fixed != text
    except ImportError:
        # Heuristik: bei „Ã"-Sequenzen UTF-8-als-Latin1 rückgängig
        if not _MOJIBAKE_HINT.search(text):
            return text, False
        try:
            fixed = text.encode("latin-1").decode("utf-8")
            return fixed, fixed != text
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text, False


def clean_text(text: str, *, drop_pua: bool = True) -> str:
    """Vollständige Normalisierung. drop_pua=False, wenn PUA-Inhalt erhalten
    bleiben soll (z. B. für CJK-Mapping-Forschung)."""
    if not text:
        return text
    # 1) Mojibake-Reparatur ZUERST — sonst zerlegt NFC die kaputten Codepunkte
    text, _ = _try_ftfy(text)
    # 2) Unicode-Normalform NFC (sehe identisch aus, sortiert sich aber)
    text = unicodedata.normalize("NFC", text)
    # 3) Ligaturen aufgelösen
    text = _LIG_RE.sub(lambda m: _LIG[m.group(0)], text)
    # 4) Replacement-Chars raus
    text = _REPLACE_CHAR.sub("", text)
    # 5) Soft Hyphens, Zero-Width, BOMs raus
    text = _INVISIBLES.sub("", text)
    # 6) PUA-Zeichen raus (optional)
    if drop_pua:
        text = _PUA.sub("", text)
    # 7) Whitespace säubern (mehrfache Spaces nach Removals)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def needs_cleaning(text: str) -> bool:
    """Schnell-Check: hat der Text mindestens eines der gesuchten Probleme?"""
    if not text:
        return False
    return bool(_LIG_RE.search(text) or _INVISIBLES.search(text)
                or _REPLACE_CHAR.search(text) or _PUA.search(text)
                or _MOJIBAKE_HINT.search(text))


if __name__ == "__main__":
    import sys
    src = sys.stdin.read()
    out = clean_text(src)
    sys.stdout.write(out)
