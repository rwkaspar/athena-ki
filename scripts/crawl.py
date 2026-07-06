#!/usr/bin/env python3
"""Athena — Crawler: aus einer Index-/Übersichtsseite die echten Dokumente ernten.

Viele eingereichte "Quellen" sind Übersichtsseiten (…/publikationen, …/themen)
ohne Sachinhalt — nur Linklisten. Dieser Crawler nimmt so eine Seite als Seed,
folgt same-domain-Links bis zu einer Tiefe, filtert Navigation/Paginierung raus
und sammelt die URLs, die echte Dokumente sind (PDFs + content_quality-geprüfte
Seiten).

Er INGESTIERT nicht selbst — er liefert eine Liste von Dokument-URLs, die dann
durch auto_review als Tier-0-Vorschläge laufen (Mensch verifiziert). So wird aus
einer wertlosen Index-Seite eine kuratierte Quellenliste.

Aufruf:
    python scripts/crawl.py <seed-url> [--depth 2] [--max-pages 60] [--json out.json]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content_quality import assess

USER_AGENT = "Athena-KI/1.0 (+https://github.com/rwkaspar/athena-ki)"
TIMEOUT = 20
DEFAULT_DEPTH = 2
DEFAULT_MAX_PAGES = 60
CRAWL_DELAY = 1.0  # Höflichkeit: 1s zwischen Requests pro Domain

# URL-Muster, die KEINE Dokumente sind (Navigation/Paginierung/Funktionsseiten)
SKIP_URL_PATTERNS = re.compile(
    r"(\?.*(page|seite|tx_solr|start|offset)=)"        # Paginierung
    r"|(/(impressum|datenschutz|kontakt|suche|search|login|sitemap|rss|feed|newsletter|ueber-uns|about|aktuelles|presse|karriere|jobs)\b)"
    r"|(/_assets?/|/assets?/|/static/|/styles?/|/scripts?/|/css/|/js/|/fileadmin/templates/)"  # Asset-Pfade
    r"|(/(en|fr|es)/)"                                  # andere Sprachversionen
    r"|(mailto:|tel:|javascript:)"
    r"|(\.(jpg|jpeg|png|gif|svg|css|js|zip|mp4|mp3|ico|woff2?|ttf)(\?|$))",
    re.I,
)

# Endung/Pfad-Teile, die nie ein Inhalts-Dokument sind (zusätzlicher Inhalts-Check)
NON_DOC_PATH = re.compile(r"/(StyleSheets|JavaScript|Resources)/|\.(css|js|map)(\?|$)", re.I)

# URL sieht nach konkretem Dokument aus (PDF oder Detail-Slug)
DOC_HINT = re.compile(r"\.pdf($|\?)|/(publikation|dokument|gutachten|bericht|studie|stellungnahme|artikel|meldung)[en]?/\S", re.I)


def _norm(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")


def _robots_ok(seed: str):
    """RobotFileParser für die Seed-Domain; .can_fetch nutzbar."""
    rp = RobotFileParser()
    base = f"{urlparse(seed).scheme}://{urlparse(seed).netloc}"
    try:
        rp.set_url(base + "/robots.txt")
        rp.read()
    except Exception:
        return None  # kein robots → erlaubt
    return rp


def _fetch(url: str) -> tuple[str, str]:
    """Liefert (content_type, text). PDFs werden als ct='pdf' erkannt, Text leer
    (Crawler bewertet PDFs allein anhand der URL als Dokument)."""
    try:
        from net_safety import safe_get  # SSRF-Schutz + geprüfte Redirects
        r = safe_get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    except Exception:
        return "", ""
    # Tote/funktionslose Seiten (404/410/5xx …) NICHT ernten und nicht weiterverfolgen.
    # Ohne diesen Check kommt z. B. eine 404-Seite als HTML zurück und ihre Links
    # (oder eine .pdf-URL mit 404) landen als vermeintliche Dokumente in der Queue.
    if r.status_code >= 400:
        return "dead", ""
    ct = r.headers.get("content-type", "").lower()
    if "pdf" in ct or url.lower().split("?")[0].endswith(".pdf"):
        return "pdf", ""
    if "html" not in ct and "text" not in ct:
        return ct, ""
    # HTML → grob zu Text
    html = r.text
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    text = re.sub(r"<[^>]+>", "\n", text)
    import html as H
    return "html", H.unescape(text), html  # type: ignore


def _links(html: str, base_url: str, domain: str) -> list[str]:
    out = []
    for href in re.findall(r'href=["\']([^"\']+)["\']', html):
        u = _norm(urljoin(base_url, href))
        p = urlparse(u)
        if p.scheme not in ("http", "https"):
            continue
        if p.netloc != domain:
            continue
        if SKIP_URL_PATTERNS.search(u):
            continue
        out.append(u)
    return out


def crawl(seed: str, depth: int, max_pages: int):
    domain = urlparse(seed).netloc
    rp = _robots_ok(seed)
    seen, documents, skipped = set(), [], []
    q = deque([(_norm(seed), 0)])
    pages = 0

    while q and pages < max_pages:
        url, d = q.popleft()
        if url in seen:
            continue
        seen.add(url)
        if rp and not rp.can_fetch(USER_AGENT, url):
            skipped.append({"url": url, "reason": "robots.txt"})
            continue

        pages += 1
        fetched = _fetch(url)
        ct = fetched[0]
        time.sleep(CRAWL_DELAY)

        if ct == "dead":
            skipped.append({"url": url, "reason": "toter Link (HTTP >= 400)"})
            continue
        if ct == "pdf":
            documents.append({"url": url, "type": "pdf", "depth": d})
            continue
        if ct != "html":
            skipped.append({"url": url, "reason": f"kein html/pdf ({ct})"})
            continue

        text = fetched[1]
        html = fetched[2] if len(fetched) > 2 else ""
        verdict = assess(text, title="")

        # Domain-Root (Homepage) ist nie ein Inhalts-Dokument
        is_root = urlparse(url).path.strip("/") == ""
        # Ist diese Seite selbst ein Dokument? (nicht der Seed — der ist die Index-Seite)
        if d > 0 and verdict["is_document"] and not NON_DOC_PATH.search(url) and not is_root:
            documents.append({"url": url, "type": "html", "depth": d,
                              "reason": verdict["reason"]})
        elif d > 0:
            skipped.append({"url": url, "reason": verdict.get("reason", "asset/non-doc")})

        # Links weiterverfolgen, solange Tiefe erlaubt
        if d < depth and html:
            for link in _links(html, url, domain):
                if link not in seen:
                    # Dokument-Hinweis-URLs priorisieren (vorne in die Queue)
                    if DOC_HINT.search(link):
                        q.appendleft((link, d + 1))
                    else:
                        q.append((link, d + 1))

    # Dedup documents nach url
    seen_doc = set()
    uniq = []
    for doc in documents:
        if doc["url"] not in seen_doc:
            seen_doc.add(doc["url"])
            uniq.append(doc)

    return {"seed": seed, "domain": domain, "pages_crawled": pages,
            "documents": uniq, "skipped_count": len(skipped)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", help="Start-/Index-URL")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    p.add_argument("--json", help="Ergebnis als JSON in Datei schreiben")
    args = p.parse_args()

    print(f"[crawl] Seed: {args.seed} (Tiefe {args.depth}, max {args.max_pages} Seiten)", file=sys.stderr)
    result = crawl(args.seed, args.depth, args.max_pages)
    print(f"[crawl] {result['pages_crawled']} Seiten besucht → "
          f"{len(result['documents'])} Dokumente, {result['skipped_count']} verworfen", file=sys.stderr)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[crawl] → {args.json}", file=sys.stderr)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
