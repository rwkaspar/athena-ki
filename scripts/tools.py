"""Athena - Tool-Use für Live-Daten.

Definiert LangChain-Tools, die Athena im Chat selbst aufrufen kann. Anders als
Ingest-Pfade fließen die Resultate NUR für die jeweils aktuelle Antwort in den
Kontext — sie werden nicht in ChromaDB persistiert.

Tier-Disziplin: Live-Fetch-Ergebnisse werden anhand der Domain via
`config/source_tiers.yaml` klassifiziert (gleiche Logik wie beim Ingest), plus
`provenance="live_fetch"` als Marker. SYSTEM-Prompt verpflichtet Athena, diese
Quellen im Output mit "(Live-Fetch, Tier X, nicht versioniert)" zu kennzeichnen.

Sicherheit: SSRF-Schutz blockt internal/loopback/Tailnet/private IP-Bereiche.
"""

import ipaddress
import os
import socket
import sys
from datetime import datetime, timezone
from urllib.parse import urlparse

import html2text
from langchain_core.tools import tool

sys.path.insert(0, os.path.dirname(__file__))

from ingest import load_source_tiers, classify_source

MAX_PAGE_BYTES = 2 * 1024 * 1024
MAX_TEXT_CHARS = 5000
NAVIGATION_TIMEOUT_MS = 30_000


def _is_blocked_host(hostname: str) -> str | None:
    """Liefert Begründungs-String wenn Host blockiert, sonst None."""
    if not hostname:
        return "leerer Hostname"
    lower = hostname.lower()
    if lower in ("localhost", "ip6-localhost"):
        return "localhost"
    try:
        ip_list = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return f"DNS-Auflösung fehlgeschlagen: {e}"
    for entry in ip_list:
        try:
            ip = ipaddress.ip_address(entry[4][0])
        except (ValueError, IndexError):
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return f"Private/Loopback-IP {ip}"
        # Tailnet CGNAT-Bereich (100.64.0.0/10) — nur für IPv4
        if isinstance(ip, ipaddress.IPv4Address) and ip in ipaddress.IPv4Network("100.64.0.0/10"):
            return f"Tailnet-IP {ip}"
    return None


@tool
def fetch_url(url: str) -> str:
    """Lädt eine URL live aus dem Internet und gibt den Textinhalt zurück.

    Verwende dieses Tool wenn die Wissensbasis keine passenden Chunks liefert
    UND die Information aus einer externen Quelle stammen muss (z. B. aktuelle
    Wetterdaten, Tagesschlagzeilen, Veranstaltungen die nicht ingestiert sind).
    Verwende es NICHT für Recht oder Gesetze — die liegen bereits ingestiert
    in der Wissensbasis vor.

    Liefert kompakten Text-Snippet mit Header der Form:
        [live_fetch] tier=N source=https://... fetched_at=ISO8601
        <Inhalt, max 5000 Zeichen>

    Die Tier-Einordnung erfolgt automatisch via Domain-Klassifikation:
    1 = Primärquelle, 2 = Medien, 3 = unklassifiziert/Kommentar.

    Bei Zugriffs-/Parse-Fehlern wird ein Fehler-String zurückgegeben, KEINE
    Exception. Du sollst dann ehrlich sagen dass der Fetch nicht ging und
    die Frage nicht auf Trainingswissen ausweichen — sag das offen.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return f"[error] Nur http/https erlaubt, nicht {parsed.scheme!r}."
    blocked = _is_blocked_host(parsed.hostname or "")
    if blocked:
        return f"[error] Host blockiert: {blocked}"

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Athena-KI/1.0 (live-fetch; +https://github.com/rwkaspar/athena-ki)"
            )
            page = context.new_page()
            try:
                response = page.goto(
                    url, wait_until="networkidle", timeout=NAVIGATION_TIMEOUT_MS
                )
            except Exception as e:
                browser.close()
                return f"[error] Konnte URL nicht laden: {e}"
            if response is None:
                browser.close()
                return "[error] Keine Antwort vom Server."
            if response.status >= 400:
                status = response.status
                browser.close()
                return f"[error] HTTP {status}"
            html = page.content()
            if len(html.encode("utf-8")) > MAX_PAGE_BYTES:
                html = html.encode("utf-8")[:MAX_PAGE_BYTES].decode(
                    "utf-8", errors="ignore"
                )
            browser.close()
    except Exception as e:
        return f"[error] Browser-Fehler: {type(e).__name__}: {e}"

    converter = html2text.HTML2Text()
    converter.ignore_images = True
    converter.body_width = 0
    text = converter.handle(html).strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n[…abgeschnitten]"

    tiers_cfg = load_source_tiers()
    tier_label, tier_rank = classify_source(url, tiers_cfg)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = (
        f"[live_fetch] tier={tier_rank} tier_label={tier_label} "
        f"source={url} fetched_at={fetched_at}\n"
    )
    return header + text


# Registry — wird vom Server verwendet
TOOLS = [fetch_url]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
