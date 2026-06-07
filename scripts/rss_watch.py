#!/usr/bin/env python3
"""
Athena RSS-Update-Pipeline.

Pollt RSS/Atom-Feeds von Tier-1-Quellen, ingestiert neue Items in den RAG.
Tracking welche Items schon ingestiert wurden via state-file.

Cron-Empfehlung: täglich 6:00 — */6h möglich für aktivere Quellen.
"""
import json
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timezone

STATE_FILE = Path.home() / ".athena" / "rss_state.json"
SCRIPTS = Path("/home/robert/athena-ki/scripts")

# Tier-1 RSS-Feeds — pro Politikfeld kuratiert (verifiziert 2026-05-28)
FEEDS = [
    # Verifizierte RSS-Feeds (Stand 2026-05-28)
    ("sipri",          "https://www.sipri.org/rss/combined.xml",                                                          1, "verteidigung"),
    ("svr_wi",         "https://www.sachverstaendigenrat-wirtschaft.de/rss.xml",                                          1, "wirtschaft"),
    ("destatis",       "https://www.destatis.de/SiteGlobals/Functions/RSSFeed/DE/RSSNewsfeed/Aktuell.xml",                1, "statistik"),
    ("bundesbank_pub", "https://www.bundesbank.de/service/rss/de/633290/feed.rss",                                        1, "wirtschaft"),
    ("bamf_news",      "https://www.bamf.de/SiteGlobals/Functions/RSS/DE/Feed/RSSNewsfeed_Pressemitteilungen.xml",        1, "migration"),
]


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def fetch_feed(url, timeout=15):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Athena-RSS/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  ❌ feed fetch: {e}", file=sys.stderr)
        return None


def parse_items(xml_text):
    """Extrahiert (link, title) aus RSS oder Atom."""
    items = []
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        print(f"  ❌ parse: {e}", file=sys.stderr)
        return items
    # RSS 2.0
    for it in root.iter("item"):
        link_el = it.find("link")
        title_el = it.find("title")
        if link_el is not None and link_el.text:
            items.append((link_el.text.strip(), (title_el.text or "").strip() if title_el is not None else ""))
    # Atom
    ns = "{http://www.w3.org/2005/Atom}"
    for it in root.iter(f"{ns}entry"):
        link_el = it.find(f"{ns}link")
        title_el = it.find(f"{ns}title")
        if link_el is not None and link_el.get("href"):
            items.append((link_el.get("href").strip(), (title_el.text or "").strip() if title_el is not None else ""))
    return items


def ingest_url(url, scope="bund", timeout=120):
    cmd = [
        sys.executable, str(SCRIPTS / "ingest.py"),
        "--url", url, "--scope", scope,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd="/home/robert/athena-ki")
        return r.returncode == 0, (r.stdout + r.stderr).decode("utf-8", errors="replace")[-500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


def main():
    state = load_state()
    new_ingest = 0
    fail_ingest = 0
    print(f"[{datetime.now(timezone.utc).isoformat()}] RSS-Watch Start")
    for name, feed_url, tier, topic in FEEDS:
        print(f"▶ {name} ({topic})")
        xml = fetch_feed(feed_url)
        if not xml:
            continue
        items = parse_items(xml)
        print(f"  {len(items)} items im Feed")
        seen = set(state.get(name, []))
        new_items = [(link, title) for link, title in items if link not in seen]
        # Max 5 neue Items pro Feed pro Lauf, um aitest nicht zu überlasten
        for link, title in new_items[:5]:
            ok, msg = ingest_url(link)
            if ok:
                seen.add(link)
                new_ingest += 1
                print(f"  ✅ {title[:60]}")
            else:
                fail_ingest += 1
                print(f"  ❌ {title[:60]} — {msg[:100]}")
            time.sleep(2)
        # State: max 200 letzte Items pro Feed merken
        state[name] = list(seen)[-200:]
        save_state(state)
    print(f"[{datetime.now(timezone.utc).isoformat()}] DONE: {new_ingest} ingestiert, {fail_ingest} fehlgeschlagen")


if __name__ == "__main__":
    main()
