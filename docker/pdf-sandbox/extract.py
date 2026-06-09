#!/usr/bin/env python3
"""PDF-Text-Extraktion im isolierten Sandbox-Container.

Liest die PDF-Bytes von STDIN, schreibt den extrahierten Text auf STDOUT.
Läuft ohne Netzwerk, read-only-Rootfs, ohne Capabilities, als nobody.
Keinerlei Host-Pfade — die Datei kommt ausschließlich über die Pipe herein,
das Original verlässt diesen Container nie wieder.

Exit-Codes:
  0  Text extrahiert (auf STDOUT)
  2  PDF defekt / nicht lesbar
  3  PDF zu groß / zu viele Seiten (Ressourcenschutz)
"""
import sys

MAX_BYTES = 25 * 1024 * 1024     # harte Obergrenze (Server limitiert ohnehin auf 10 MB)
MAX_PAGES = 500
MAX_CHARS = 2_000_000            # ~ genug für sehr lange Dokumente, bremst Text-Bomben


def main() -> int:
    data = sys.stdin.buffer.read(MAX_BYTES + 1)
    if not data:
        print("leere Eingabe", file=sys.stderr)
        return 2
    if len(data) > MAX_BYTES:
        print("PDF zu groß", file=sys.stderr)
        return 3

    import io
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, Exception) as e:  # noqa: BLE001 — Sandbox: jeder Parserfehler = unbrauchbar
        print(f"PDF nicht lesbar: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    n_pages = len(reader.pages)
    if n_pages > MAX_PAGES:
        print(f"zu viele Seiten ({n_pages} > {MAX_PAGES})", file=sys.stderr)
        return 3

    out = []
    total = 0
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — einzelne defekte Seite überspringen
            t = ""
        if t:
            out.append(t)
            total += len(t)
            if total >= MAX_CHARS:
                out.append("\n[... Text abgeschnitten ...]")
                break

    sys.stdout.write("\n\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
