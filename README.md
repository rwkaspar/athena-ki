# Athena KI

**Athena** ist ein KI-Beratungsinstrument für faktenbasierte politische Analyse —
evidenzbasiert, überparteilich, transparent.

## Vision und Pilot-Scope

Fernziel: ein Werkzeug für die geplante Partei **EVIDENZ** (Ein Volk,
Informiert und Demokratisch Entschieden, Nicht Zufällig) auf Bundesebene.
Die KI soll Faktenlage, Rechtsrahmen, Optionen und Trade-offs strukturiert
aufbereiten — die normative Entscheidung bleibt beim Menschen.

**Aktueller Pilot-Scope: Gemeinde Pfofeld (91378), Landkreis
Weißenburg-Gunzenhausen, Bayern.** Pfofeld ist bewusst gewählt: kleine
Domäne, klarer Rechtsrahmen (Bayerische Gemeindeordnung), überschaubare
Quellenlage, echte Vergleichsfälle in Nachbargemeinden. Die Methodik
wird hier validiert, bevor sie auf Bundesebene skaliert wird.

## Methodisches Kernprinzip

Athena liefert **strukturierte Optionsanalyse**, nicht "die beste Lösung".
Zu jedem Thema werden geliefert:

- belastbare Faktenlage mit Quellen und Verifikationsstatus
- relevanter Rechtsrahmen
- mehrere Lösungsoptionen mit expliziten Trade-offs
- zugrundeliegende Wertannahmen jeder Option
- empirische Evidenz aus Vergleichsfällen

Wer eine Empfehlung braucht, soll sie sich aus den Optionen selbst bilden.

## Architektur

| Komponente | Aufgabe |
|---|---|
| **Orchestrierungs-VM** | Python-Pipeline, ChromaDB, Web-Scraping (Ubuntu/Debian, ~6 vCPU / 16 GB RAM / 100 GB SSD) |
| **Ollama-Server** (separat) | LLM-Inference, im privaten Netz erreichbar (Tailscale/WireGuard) |
| **LLM** | Qwen3.6 35B-A3B (MoE) als Base, custom Modell `athena` mit Persona-Prompt |
| **Embeddings** | `nomic-embed-text` |
| **Vektor-DB** | ChromaDB |
| **RAG-Framework** | LangChain |

Die Trennung VM ↔ Inference-Server ist beabsichtigt: Inference braucht viel
RAM/Bandbreite und sollte auf dafür ausgelegter Hardware laufen, die
Orchestrierungs-VM bleibt schlank.

## Setup

### Voraussetzungen

- Eine Linux-VM für die Orchestrierung (Python 3.10+).
- Eine separate Maschine mit installiertem [Ollama](https://ollama.com)
  und ausreichend RAM für das Base-Modell (~25 GB für Qwen3.6 35B-A3B in Q4).
  Kann auch dieselbe Maschine sein, dann entfällt der Remote-Setup.

### 1. Repository klonen

```bash
git clone https://github.com/rwkaspar/athena-ki.git
cd athena-ki
```

### 2. Modelle auf dem Ollama-Server bereitstellen

Auf dem Inference-Host (oder lokal, falls alles auf einer Maschine):

```bash
ollama pull qwen3.6:35b-a3b
ollama pull nomic-embed-text
ollama create athena -f Modelfile
```

### 3. Remote-Inference (optional)

Wenn Ollama auf einem anderen Host läuft, muss der Service auf einem
nicht-localhost-Interface lauschen. Per systemd-Override:

```ini
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Dann `sudo systemctl daemon-reload && sudo systemctl restart ollama`.
Auf der Orchestrierungs-VM:

```bash
export OLLAMA_HOST=http://<inference-host>:11434
```

Achtung: Ollama hat keine Authentifizierung. Den Port nur in einem
privaten Netz (VPN/Tailscale/WireGuard) erreichbar machen, nicht
öffentlich.

### 4. Python-Umgebung

```bash
python3 -m venv athena-env
source athena-env/bin/activate
pip install -r requirements.txt
```

### 5. Wissensbasis befüllen

```bash
cd scripts
python ingest.py --url https://www.gesetze-bayern.de/Content/Document/BayGO/true
python ingest.py --dir ../documents/pfofeld/
```

### 6. Athena befragen

```bash
python query.py "Welche Voraussetzungen gelten für eine Bürgerversammlung nach Art. 18 BayGO?"
python query.py --interactive
```

## Projektstruktur

```
athena-ki/
├── claude.md              # Projekt-Kontext, Methodik, Tech-Topologie
├── Modelfile              # Ollama Custom Model (Persona-Prompt)
├── requirements.txt       # Python-Abhängigkeiten
├── scripts/
│   ├── ingest.py          # Dokumente in Wissensbasis einspeisen
│   ├── query.py           # Wissensbasis abfragen
│   └── generate_post.py   # Stellungnahmen generieren (in Überarbeitung)
├── prompts/
│   └── system_prompt.txt  # archivierter Bundes-Prompt — nicht aktiv
├── documents/             # Rohdokumente (gitignored)
├── athena-db/             # ChromaDB (gitignored)
└── output/posts/          # Generierte Stellungnahmen (gitignored)
```

## Aktueller Stand

- [x] Custom Athena-Modell auf Qwen3.6 35B-A3B-Basis
- [x] Remote-Inference über Ollama im privaten Netz
- [x] End-to-end-RAG funktional (klassischer Single-Pass)
- [ ] Tier-Metadaten beim Ingest (Primär-/Sekundär-/Kommentarquellen)
- [ ] Hybrid-RAG (statische Rechtsbasis getrennt von tagesaktuellen Quellen)
- [ ] Strukturierte Optionsanalyse mit JSON-Schema
- [ ] Critique-Pass (Devil's-Advocate-Stufe)
- [ ] Notion-Integration als Doku-Layer
- [ ] EVIDENZ-Skalierung auf Bundesebene

## Lizenz

MIT

## Mitmachen

Athena ist Open Source. Beiträge sind willkommen — Transparenz ist Kernprinzip
des Projekts. Bevorzugt sind Beiträge zur Methodik (Pipeline-Stufen,
Quellen-Hierarchie), zur Wissensbasis kommunaler Primärquellen und zur
Test-Infrastruktur.
