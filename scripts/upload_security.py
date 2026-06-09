"""Upload-Härtung für Quelleneinreichungen.

Drei Schutzschichten, bevor eine hochgeladene Datei den Server berührt:

1. VIRENSCAN — die rohen Bytes werden per ClamAV (clamd, INSTREAM über TCP)
   gescannt. Funde werden abgelehnt, die Datei nie persistiert.
2. SANDBOX-PARSING — PDFs werden NICHT im Serverprozess geparst, sondern in
   einem wegwerfbaren Docker-Container ohne Netzwerk, read-only-Rootfs, ohne
   Capabilities, als nobody. Selbst ein Exploit im PDF-Parser bleibt dort.
3. PDF→TEXT-ONLY — aus PDFs wird nur der Text extrahiert; das Original wird
   verworfen und nie gespeichert. Damit landet kein binärer, potenziell aktiver
   Inhalt in der Wissensbasis oder im Dateisystem.

Konfiguration (Env):
  ATHENA_CLAMD_HOST       (default 127.0.0.1)
  ATHENA_CLAMD_PORT       (default 3310)
  ATHENA_CLAMD_TIMEOUT_S  (default 30)
  ATHENA_PDF_SANDBOX_IMAGE(default athena-pdf-sandbox:latest)
  ATHENA_SANDBOX_TIMEOUT_S(default 60)
  ATHENA_REQUIRE_SCAN     (default 1 = fail-closed: ist clamd nicht erreichbar,
                           wird der Upload abgelehnt statt ungescannt akzeptiert)
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess

CLAMD_HOST = os.getenv("ATHENA_CLAMD_HOST", "127.0.0.1")
CLAMD_PORT = int(os.getenv("ATHENA_CLAMD_PORT", "3310"))
CLAMD_TIMEOUT_S = float(os.getenv("ATHENA_CLAMD_TIMEOUT_S", "30"))
SANDBOX_IMAGE = os.getenv("ATHENA_PDF_SANDBOX_IMAGE", "athena-pdf-sandbox:latest")
SANDBOX_TIMEOUT_S = float(os.getenv("ATHENA_SANDBOX_TIMEOUT_S", "60"))
REQUIRE_SCAN = os.getenv("ATHENA_REQUIRE_SCAN", "1") not in ("0", "false", "False", "")


class ScanError(RuntimeError):
    """clamd nicht erreichbar o. Ä. — bei REQUIRE_SCAN führt das zur Ablehnung."""


class InfectedError(RuntimeError):
    """Datei enthält eine erkannte Bedrohung."""
    def __init__(self, signature: str):
        self.signature = signature
        super().__init__(f"Schadsoftware erkannt: {signature}")


class SandboxError(RuntimeError):
    """PDF konnte im Sandbox-Container nicht (sicher) geparst werden."""


# ---------------------------------------------------------------- Virenscan ---
def _clamd_instream(data: bytes) -> str:
    """Bytes per INSTREAM an clamd schicken, Roh-Antwort zurückgeben."""
    try:
        s = socket.create_connection((CLAMD_HOST, CLAMD_PORT), timeout=CLAMD_TIMEOUT_S)
    except OSError as e:
        raise ScanError(f"clamd nicht erreichbar ({CLAMD_HOST}:{CLAMD_PORT}): {e}") from e
    try:
        s.settimeout(CLAMD_TIMEOUT_S)
        s.sendall(b"zINSTREAM\0")
        CHUNK = 8192
        for i in range(0, len(data), CHUNK):
            chunk = data[i:i + CHUNK]
            s.sendall(struct.pack("!L", len(chunk)) + chunk)
        s.sendall(struct.pack("!L", 0))  # Null-Chunk = Ende
        resp = b""
        while True:
            buf = s.recv(4096)
            if not buf:
                break
            resp += buf
            if b"\0" in buf:
                break
        return resp.decode(errors="replace").strip("\0\n ")
    except OSError as e:
        raise ScanError(f"clamd-Kommunikation fehlgeschlagen: {e}") from e
    finally:
        try:
            s.close()
        except OSError:
            pass


def scan_bytes(data: bytes) -> None:
    """Rohe Upload-Bytes scannen. Wirft InfectedError bei Fund, ScanError wenn
    clamd nicht erreichbar ist UND REQUIRE_SCAN gesetzt ist. Kein Rückgabewert:
    kehrt nur sauber zurück, wenn die Datei als unbedenklich gilt."""
    try:
        resp = _clamd_instream(data)
    except ScanError:
        if REQUIRE_SCAN:
            raise
        return  # fail-open nur, wenn explizit so konfiguriert
    # Antwortformate: "stream: OK"  /  "stream: <Sig> FOUND"  /  "... ERROR"
    if resp.endswith("FOUND"):
        sig = resp.split(":", 1)[-1].strip()
        if sig.endswith("FOUND"):
            sig = sig[:-len("FOUND")].strip()
        raise InfectedError(sig or "unbekannt")
    if "ERROR" in resp:
        if REQUIRE_SCAN:
            raise ScanError(f"clamd meldete Fehler: {resp}")
        return
    # alles andere (insb. "stream: OK") = sauber


def clamd_available() -> bool:
    """PING an clamd — für Health-Checks."""
    try:
        s = socket.create_connection((CLAMD_HOST, CLAMD_PORT), timeout=3)
    except OSError:
        return False
    try:
        s.sendall(b"zPING\0")
        return b"PONG" in s.recv(64)
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


# ------------------------------------------------------- Sandbox-PDF-Parsing ---
def extract_pdf_text(data: bytes) -> str:
    """PDF-Bytes in einem isolierten Docker-Container zu reinem Text parsen.

    Der Container hat KEIN Netzwerk, read-only-Rootfs, keine Capabilities,
    läuft als nobody und bekommt die Datei nur über stdin. Das Original wird
    danach von der aufrufenden Stelle verworfen — nur der Text bleibt."""
    cmd = [
        "docker", "run", "--rm", "-i",
        "--network", "none",
        "--read-only",
        "--user", "65534:65534",
        "--memory", "512m", "--memory-swap", "512m",
        "--pids-limit", "64",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:size=16m",
        SANDBOX_IMAGE,
    ]
    try:
        proc = subprocess.run(
            cmd, input=data, capture_output=True,
            timeout=SANDBOX_TIMEOUT_S,
        )
    except FileNotFoundError as e:
        raise SandboxError("docker nicht verfügbar für Sandbox-Parsing") from e
    except subprocess.TimeoutExpired as e:
        raise SandboxError("Sandbox-Parsing hat das Zeitlimit überschritten") from e

    if proc.returncode != 0:
        err = (proc.stderr or b"").decode(errors="replace").strip()[:300]
        raise SandboxError(f"PDF nicht verarbeitbar (rc={proc.returncode}): {err}")

    text = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
    if not text:
        raise SandboxError("PDF enthielt keinen extrahierbaren Text "
                           "(evtl. reines Scan-Bild ohne OCR).")
    return text
