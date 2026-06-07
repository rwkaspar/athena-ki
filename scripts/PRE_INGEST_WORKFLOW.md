# Pre-Ingest-Workflow für Politikfeld-Analysen

Vor jeder Athena-Analyse für ein neues Politikfeld diese drei Schritte machen, um Halluzinations-Risiko zu minimieren.

## 1. Themenspezifische URL-Liste zusammenstellen (10-20 URLs)

Drei Säulen abdecken:

### a) Faktenbasis (Tier-1)
- Amtliche Statistik (Destatis, BA, Eurostat) — spezifische Themen-Seiten, nicht nur Hauptseite
- Sachverständigenrats-Gutachten zum Thema
- Aktuelle Bundesregierung-/Ministerien-Berichte
- BVerfG-Entscheidungen volltext (wenn rechtsrelevant)
- EU-Recht via eur-lex (Verordnungen, Richtlinien, Vorlagebeschlüsse EuGH)

### b) Wissenschaftliche Forschung (Tier-1)
- Mindestens 2 verschiedene Forschungs-Institute (DIW + ifo + ZEW für Wirtschaftsfragen, SWP + RUSI + SIPRI für Sicherheit, etc.) — Triangulation
- IAB-Studien für Arbeits-/Sozialfragen
- IPCC / ERK für Klima
- Aktuelle Top-Quellen aus den letzten 12 Monaten zum Thema

### c) Internationale Vergleichsfälle (Tier-1)
- OECD-Berichte
- Konkrete Länder-Quellen (z. B. INSEE für FR, OFS für CH, SCB für SE)
- Vergleichbare Reformen mit Empirie

## 2. Ingestieren

```bash
cd /home/robert/athena-ki
source athena-env/bin/activate

# Eine URL
python scripts/ingest.py --url https://www.example.de/bericht.pdf --scope bund

# Tier-Override (wenn URL nicht in source_tiers_bund.yaml ist)
python scripts/ingest.py --url URL --scope bund --tier 1 --source-label "SVR Jahresgutachten 2025"

# PDF-Datei lokal
python scripts/ingest.py --file ~/downloads/bericht.pdf --scope bund

# Verzeichnis
python scripts/ingest.py --dir ~/sources/migration/ --scope bund

# Für JS-gerenderte Seiten
python scripts/ingest.py --url URL --render --wait-selector "#main-content" --scope bund
```

## 3. Ingest verifizieren vor Athena-Lauf

```bash
# Domains im Bund-RAG prüfen
python3 -c "
import sqlite3, urllib.parse
db = sqlite3.connect('/home/robert/athena-ki/athena-db/chroma.sqlite3')
c = db.cursor()
c.execute(\"SELECT string_value FROM embedding_metadata WHERE key='source';\")
domains = {}
for (s,) in c.fetchall():
    if s:
        d = urllib.parse.urlparse(s).netloc.lower().removeprefix('www.')
        domains[d] = domains.get(d, 0) + 1
for d in sorted(domains): print(f'{domains[d]:5d}  {d}')
"

# Probe-Query an Athena
curl -sS -X POST http://100.105.70.24:8765/chat \
  -H "Content-Type: application/json" \
  -d '{"scope":"bund","provider":"mistral","message":"Welche Quellen hast du zu THEMA X?"}'
```

## Update-Pipeline

`scripts/rss_watch.py` pollt RSS-Feeds wichtiger Tier-1-Quellen und ingestiert
neue Items automatisch. Empfohlen als Cron-Job (siehe Kommentar im Script).

State-Tracking in `~/.athena/rss_state.json` verhindert Doppel-Ingest.

## Re-Crawl bestehender Quellen

Berichte werden regelmäßig aktualisiert (ERK-Berichte, ifo Konjunkturprognose).
Quellen-URLs alle 90 Tage neu ingestieren — Chunks werden über die `source`-URL
deduplikiert, wenn ChromaDB die ID-Logik so erkennt.

Manuelles Re-Ingest:

```bash
# z. B. Klima-Quellen erneut crawlen
for url in $(grep -E "expertenrat-klima|umweltbundesamt|agora-" /tmp/ingest_urls.txt); do
  python scripts/ingest.py --url "$url" --scope bund
  sleep 2
done
```
