"""Zentraler SSRF-Schutz für alle server-seitigen URL-Fetches.

EIN Ort für die Host-/IP-Blockliste, damit Submission-, Auto-Review-, Ingest-,
Crawl- und Tool-Pfad denselben Schutz nutzen (vorher hatte nur das fetch_url-Tool
einen Filter). Blockt loopback/private/link-local/multicast/reserved/unspecified,
den IPv4-Tailnet-CGNAT-Bereich (100.64.0.0/10) und IPv6-ULA (fc00::/7, deckt den
Tailscale-ULA fd7a::/48 ab).

Keine Projekt-Importe → von jedem Modul importierbar, ohne Zirkelbezug.
Restrisiko: keine IP-Pinnung, daher bleibt ein schmales DNS-Rebinding-Fenster
zwischen Prüfung und Verbindung; Redirects werden in safe_get pro Hop neu geprüft.
"""
import ipaddress
import socket
from urllib.parse import urljoin, urlparse

_TAILNET_V4 = ipaddress.IPv4Network("100.64.0.0/10")
_ULA_V6 = ipaddress.IPv6Network("fc00::/7")


class BlockedURLError(Exception):
    """Wird geworfen, wenn eine URL aus SSRF-Gründen nicht abgerufen werden darf."""


def is_blocked_host(hostname: str) -> str | None:
    """Begründungs-String, wenn der Host blockiert ist, sonst None."""
    if not hostname:
        return "leerer Hostname"
    if hostname.lower() in ("localhost", "ip6-localhost"):
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
        if (ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            return f"Private/Loopback-IP {ip}"
        if isinstance(ip, ipaddress.IPv4Address) and ip in _TAILNET_V4:
            return f"Tailnet-IP {ip}"
        if isinstance(ip, ipaddress.IPv6Address) and ip in _ULA_V6:
            return f"IPv6-ULA {ip}"
    return None


def assert_url_safe(url: str) -> None:
    """Wirft BlockedURLError, wenn Schema ≠ http/https oder der Host blockiert ist."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https"):
        raise BlockedURLError(f"nur http/https erlaubt, nicht {parsed.scheme!r}")
    reason = is_blocked_host(parsed.hostname or "")
    if reason:
        raise BlockedURLError(reason)


def safe_get(url, *, max_redirects=5, **kwargs):
    """requests.get mit SSRF-Schutz: prüft das Ziel VOR jedem Request und folgt
    Redirects manuell — jedes Redirect-Ziel wird neu gegen is_blocked_host geprüft
    (verhindert den Redirect-Bypass des Filters). Wirft BlockedURLError, wenn ein
    Ziel blockiert ist; sonst wie requests.get."""
    import requests
    kwargs["allow_redirects"] = False
    kwargs.setdefault("timeout", 30)
    current = (url or "").strip()
    for _ in range(max_redirects + 1):
        assert_url_safe(current)
        resp = requests.get(current, **kwargs)
        if resp.is_redirect or resp.is_permanent_redirect:
            loc = resp.headers.get("Location")
            if not loc:
                return resp
            current = urljoin(current, loc)
            continue
        return resp
    raise BlockedURLError("zu viele Redirects")
