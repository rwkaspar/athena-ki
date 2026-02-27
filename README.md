# 🏛️ Athena KI

**Athena** ist eine KI-gestützte politische Persönlichkeit für Deutschland – evidenzbasiert, überparteilich, transparent und nicht korrumpierbar.

## Vision

Athena ist eine politische Bewegung, die eine KI als Beratungsinstanz nutzt, vertreten durch eine menschliche Person. Ziel ist es, Korruption in der Politik zu bekämpfen, indem politische Entscheidungen rein auf Fakten, Daten und Evidenz basieren – frei von persönlichen Interessen, Lobbyismus und Ideologie.

## Technischer Stack

| Komponente | Technologie |
|---|---|
| LLM | Qwen 2.5 32B (via Ollama) |
| Embedding | nomic-embed-text (via Ollama) |
| Vektordatenbank | ChromaDB |
| RAG-Framework | LangChain |
| Infrastruktur | Proxmox LXC Container |

## Setup

### Voraussetzungen

- Server mit mindestens 32GB RAM (empfohlen: 48GB+)
- [Ollama](https://ollama.com) installiert
- Python 3.10+

### 1. Repository klonen

```bash
git clone https://github.com/rwkaspar/athena-ki.git
cd athena-ki
```

### 2. Qwen & Embedding-Modell laden

```bash
ollama pull qwen2.5:32b
ollama pull nomic-embed-text
```

### 3. Athena-Modell erstellen

```bash
ollama create athena -f Modelfile
```

### 4. Python-Umgebung einrichten

```bash
python3 -m venv athena-env
source athena-env/bin/activate
pip install -r requirements.txt
```

### 5. Wissensbasis befüllen

```bash
cd scripts
python ingest.py --url https://www.gesetze-im-internet.de/gg/
```

### 6. Athena befragen

```bash
python query.py "Was sagt das Grundgesetz zur Menschenwürde?"
python query.py --interactive
```

### 7. Social Media Posts generieren

```bash
python generate_post.py "Die Bundesregierung plant eine Erhöhung der Mehrwertsteuer"
python generate_post.py --topic "Rentenpolitik" --platform twitter
```

## Projektstruktur

```
athena-ki/
├── Modelfile              # Ollama Custom Model (Qwen 2.5 32B + System-Prompt)
├── requirements.txt       # Python-Abhängigkeiten
├── scripts/
│   ├── ingest.py          # Dokumente in Wissensbasis einspeisen
│   ├── query.py           # Athena mit RAG befragen
│   └── generate_post.py   # Social Media Content generieren
├── prompts/
│   └── system_prompt.txt  # Athenas Persönlichkeit & Verhalten
├── documents/             # Rohdokumente (nicht im Repo)
├── athena-db/             # ChromaDB Vektordatenbank (nicht im Repo)
└── output/
    └── posts/             # Generierte Stellungnahmen
```

## Athenas Grundprinzipien

- **Evidenzbasiert** – Politik auf Basis von Daten, Forschung und Fakten
- **Überparteilich** – keine Einordnung ins politische Spektrum
- **Transparent** – jede Entscheidung wird öffentlich begründet
- **Anti-Korruption** – keine Lobbyabhängigkeiten, keine verdeckten Interessen
- **Direkt** – klare, verständliche Kommunikation für alle Bürger

## Roadmap

- [x] Qwen 2.5 32B als Basis-LLM
- [x] System-Prompt & Persönlichkeit
- [x] RAG-System mit ChromaDB
- [ ] Wissensbasis befüllen (Grundgesetz, Koalitionsvertrag, Haushaltsdaten)
- [ ] Social Media Präsenz aufbauen
- [ ] Echtzeit-Datenanbindung (Nachrichten, Statistiken)
- [ ] Faktenprüfungs-Modul
- [ ] Bürger-Chatbot

## Lizenz

MIT

## Mitmachen

Athena ist ein Open-Source-Projekt. Beiträge sind willkommen! Transparenz ist unser Kernprinzip – deshalb ist der gesamte Code öffentlich.
