# Athena — Projekt-Kontext für Claude Code

## Was ist Athena?

Athena ist ein KI-Beratungsinstrument für faktenbasierte politische Analyse.
Fernziel: Kopplung an die geplante Partei EVIDENZ (Ein Volk, Informiert und
Demokratisch Entschieden, Nicht Zufällig) auf Bundesebene.

**Aktueller Pilot-Scope: Gemeinde Pfofeld (91378), Landkreis Weißenburg-
Gunzenhausen, Bayern.** Pfofeld ist bewusst gewählt: kleine Domäne, klarer
Rechtsrahmen (Bayerische Gemeindeordnung), überschaubare Quellenlage,
echte Vergleichsfälle in Nachbargemeinden. Die Methodik wird hier validiert,
bevor sie auf Bundesebene skaliert wird.

Athena analysiert kommunalpolitische Themen strukturiert. Die normative
Entscheidung bleibt beim Menschen — beim Gemeinderat, beim Bürgermeister,
bei den Bürgern.

## Methodisches Kernprinzip — nicht verhandelbar

Athena liefert **strukturierte Optionsanalyse**, nicht "die beste Lösung".
Zu jedem Thema werden geliefert:

- belastbare Faktenlage mit Quellen und Verifikationsstatus
- relevanter Rechtsrahmen (BayGO, Satzungen, einschlägige Bundes-/Landesgesetze)
- 2–4 Lösungsoptionen mit expliziten Trade-offs
- zugrundeliegende Wertannahmen jeder Option
- empirische Evidenz aus Vergleichsfällen (Nachbargemeinden, ähnliche Kommunen)

Wenn ein Prompt, eine Funktion oder ein Output "die beste Lösung", "klare
Position" oder "Empfehlung" produziert, ist das ein Bug. Bestehende
Formulierungen dieser Art (u.a. in `scripts/generate_post.py` und im
`Modelfile`) sind beim nächsten Refactor zu ersetzen, nicht zu reproduzieren.

Die methodischen Prinzipien sind scope-unabhängig. Sie gelten für Pfofeld
genauso wie später für Bundesebene. Der Scope-Wechsel wird primär eine Frage
der Wissensbasis und der Quellen-Hierarchie sein, nicht der Methodik.

## Quellen-Hierarchie (Pilot-Scope Pfofeld)

- **Tier 1 (Primärquellen kommunal):** Bayerische Gemeindeordnung,
  Pfofelder Satzungen und Verordnungen, Gemeinderatsprotokolle und
  -beschlüsse, Haushaltspläne der Gemeinde, Bebauungspläne,
  Landratsamtsbescheide, einschlägige Bundes-/Landesgesetze
  (BauGB, BayBO, KAG)
- **Tier 1 (Primärquellen statistisch):** Bayerisches Landesamt für
  Statistik, Daten des Landkreises, amtliche Tourismusstatistiken
  Fränkisches Seenland
- **Tier 2 (regionale Medien):** Altmühl-Bote, Weißenburger Tagblatt,
  Bayerischer Rundfunk Mittelfranken, Nordbayerischer Kurier
- **Tier 2 (überregional, falls relevant):** SZ, FAZ, Tagesschau —
  nur bei Themen mit überregionalem Bezug
- **Tier 3 (Kommentar/lokal):** Vereinsmitteilungen, Leserbriefe,
  Bürgerinitiativen-Positionen, kommunalpolitische Blogs

Bei kommunalpolitischen Fragen sind Tier-1-Quellen oft nicht digital
verfügbar. Das ist ein realer Constraint des Pilot-Scopes, der
sauber dokumentiert werden muss — nicht durch Tier-2-Surrogate verdeckt.

Jeder Chunk bekommt Tier-Metadaten beim Ingest. Tier-3-Quellen werden im
Output transparent als solche markiert. Claims werden bevorzugt gegen
Tier-1-Quellen verifiziert.

## Ziel-Pipeline (sechs deterministische Stufen)

1. **Scoping** — User definiert präzises Thema (nicht "Tourismus", sondern
   z.B. "Soll Pfofeld saisonale Parkgebühren am Kleinen Brombachsee einführen?")
2. **Source Aggregation** — tiered, mit Metadaten — *implementiert*
3. **Claim Extraction & Verification** — atomare Aussagen, gegen Tier-1
   geprüft, Status pro Claim — *implementiert* (`scripts/verify.py`,
   default `gemma3:27b`). Setzt `verification_status` (verifiziert /
   teilweise / nicht_belegt / widersprochen) und `evidence_quote` pro
   Faktum. Caveat: Modell antwortete auf "JSON-Array"-Prompt nur mit
   dem ersten Element — Wrapper-Objekt `{verifications: [...]}` mit
   automatischem Unwrap im Parser löst das.
4. **Structured Analysis** — JSON nach festem Schema (siehe Kernprinzip).
   Schema liegt als Pydantic-Modell in `scripts/schema.py`. **Zwei-Phasen-
   Implementierung:** Athena produziert weiterhin strukturiertes Markdown
   (Modelfile-SYSTEM erzwingt die 6 Stufen), und `scripts/structure.py`
   konvertiert das Markdown via separater Strukturierungs-LLM-Aufruf
   (`format="json"` + Pydantic-Validation, ein Retry bei Schema-Fehler) in
   eine `Optionsanalyse`-Instanz. Default-Modell: `gemma3:27b` (env-Var
   `STRUCTURE_MODEL`). Aktiviert per `query.py --structure`. Hintergrund:
   Ollamas `format=<dict-schema>` in 0.18.3 erzwingt das Schema nicht
   strikt; Post-Processing ist robuster.
5. **Critique Pass** — zweiter LLM-Lauf in Devil's-Advocate-Rolle —
   *implementiert* (`scripts/critique.py`, default `gemma3:27b`)
6. **Documentation Sink** — Notion-Page mit Quellenliste und Verifikationsstatus
   — *in Arbeit*

Jede Stufe ist einzeln testbar. Niemals als ein einziger agentischer Loop bauen.

## Beispielhafte Pilot-Themen (zum Testen der Pipeline)

- Tourismus-Management am Kleinen Brombachsee (Parkraum, Gebühren,
  Saisonbelastung)
- Wasserversorgung und -gebühren (KAG-relevant)
- Baugebiet-Ausweisung in einem der Ortsteile
- Glasfaserausbau / Breitbandförderung
- ÖPNV-Anbindung Richtung Gunzenhausen / Weißenburg
- Energetische Sanierung kommunaler Gebäude

Das sind reale Entscheidungssituationen einer 1.500-Einwohner-Gemeinde
und damit gut bemessen für End-to-End-Pipeline-Tests.

## Hybrid-RAG-Architektur

- **Static collection** (ChromaDB): Rechtsgrundlagen, Satzungen, periodisch
  re-indexiert (monatlich)
- **Fresh collection** (ChromaDB): aktuelle Berichte, Gemeinderatsprotokolle,
  on-demand re-indexiert
- Collections nicht vermischen — Tagesgeschehen darf die Rechtsbasis nicht
  kontaminieren

## Tech-Stack & Deployment-Topologie

- **VM** (Debian/Ubuntu LTS): Python-Pipeline, ChromaDB, Web-Scraping —
  Empfehlung 6–8 vCPU, 16 GB RAM, 100 GB SSD; LXC-Container ist okay.
  Aktueller Stand: Ubuntu 24.04 KVM-VM mit 6 vCPU / 31 GB RAM / 100 GB SSD
  (98 GB nutzbar via LVM).
- **Ollama-Server** (separate Box im privaten Tailscale-Mesh): Inferenz läuft
  hier, NICHT auf der VM. Erreichbar unter `100.101.225.56`, Hostname `aitest`,
  SSH-User `robert`, Key hinterlegt. Aktuell ist das ein SOYO Mini-PC mit
  AMD Ryzen 7 H 255 (Hawk Point, DDR5 SO-DIMM, ~90–120 GB/s Bandbreite) und
  32 GB RAM — Stopgap, deutlich knapp für 30B+-Modelle, nicht die finale
  Phase-2-Hardware. Phase-2-Ziel bleibt die HP Z2 Mini G1a (Strix Halo,
  128 GB LPDDR5X @ 256 GB/s). Die Box ist multi-tenant: andere Modelle
  (Alfred etc.) liegen parallel drauf, also bei Service-Restarts und RAM-
  Druck Rücksicht nehmen.
- **LangChain + ChromaDB + bge-m3** für RAG (Embedding-Modell ist über
  env-Var `EMBEDDING_MODEL` umstellbar; Default `bge-m3`, multilingual,
  1024-dim, deutlich stärker auf Deutsch als das vorherige `nomic-embed-text`).
  Beim Wechsel des Embedding-Modells: ChromaDB komplett neu aufbauen, weil
  Dimensionen pro Collection fix sind.
- **Modelle:** aktuell `qwen3.6:35b-a3b` (MoE) als Athena-Base, `gemma3:27b`
  für Critique/Structure/Verify, `bge-m3` für Embeddings. Phase-2-Ziel:
  `qwen3-235b-a22b` (MoE) auf der Strix-Halo-Hardware mit 128 GB.
- **Notion** als Doku-Layer (MCP-Integration)
- Ollama-Connection IMMER über `OLLAMA_HOST` env var, niemals hardgecodet

## Aktueller Repo-Stand

- `Modelfile` — Pfofeld-Bürgermeisterin, gültige Identität für den Pilot.
  Base ist `qwen3.6:35b-a3b` (MoE, ~23 GB Q4_K_M). SYSTEM-Prompt ist auf
  das Optionsanalyse-Kernprinzip ausgerichtet: Athena liefert die 6-Stufen-
  Struktur (Faktenlage → Rechtsrahmen → 2–4 Optionen mit Trade-offs →
  Wertannahmen → Vergleichsfälle → Offene Fragen/Konfidenz) und trifft
  explizit keine Entscheidungen. Wissensfragen werden weiterhin faktisch
  beantwortet — die Optionsanalyse-Pflicht greift nur bei Entscheidungs-
  fragen. Modell muss bei jeder Modelfile-Änderung auf dem Remote neu
  gebaut werden: `scp Modelfile robert@100.101.225.56:/tmp/ &&
  ssh robert@100.101.225.56 'ollama create athena -f /tmp/Modelfile'`.
- `prompts/system_prompt.txt` — bundesweite Skizze als Fernziel archiviert,
  **nicht aktiv**. Nicht parallel pflegen — Modelfile ist die Quelle der Wahrheit.
- `scripts/ingest.py`, `scripts/query.py`, `scripts/generate_post.py` —
  Ollama-Connection läuft sauber über `OLLAMA_HOST` env-Var (default
  `http://localhost:11434`). Für Remote-Inference:
  `export OLLAMA_HOST=http://100.101.225.56:11434`. Alle Generation-LLM-
  Aufrufe nutzen `reasoning=False`, weil qwen3.6's Thinking-Mode mit
  langchain-ollama-Streaming nicht zuverlässig ins `response`-Feld
  zurückschreibt (Antworten landen manchmal komplett im `thinking`-Feld
  und kommen als leerer String zurück).
- `scripts/ingest.py` — Tier-Metadaten via `config/source_tiers.yaml`
  (domain-basierte Klassifikation, Tier 1/2/3 plus `source_type`
  static/fresh, `ingested_at`-Timestamp). CLI-Override per
  `--tier <N> --source-label <name>` für File-Ingests ohne Domain.
  Für JS-gerenderte Seiten (z. B. `gesetze-bayern.de`) gibt es
  `--render` — lädt die Seite über Playwright/Headless-Chromium und
  konvertiert via `html2text`. Optional `--wait-selector <css>` zum
  Warten auf einen DOM-Knoten, sonst wird auf `networkidle` gewartet.
  Routing: Chunks mit `source_type=static` (Tier 1) gehen in die
  Collection `static`, alle anderen in `fresh`. Beide Collections
  liegen im selben `athena-db/`-Persistdir.
- `scripts/retrieval.py` — geteilte Retrieval-Bausteine: Multi-Collection-Setup
  (`get_vectorstores`, `COLLECTION_NAMES = ["static", "fresh"]`),
  `collection_for_source_type` für ingest-Routing, `tier_aware_retrieve` zieht
  Kandidaten aus *beiden* Collections und re-rankt sie tier-gewichtet.
  Boost-Werte (`{1: 1.0, 2: 0.75, 3: 0.5}`) hier im Code, nicht in der YAML —
  sie sind LLM-/Domain-Tuning, nicht Quellen-Pflege.
- `scripts/critique.py` — Stage 5 der Pipeline (Critique-Pass). Eigener LLM-Aufruf
  mit Devil's-Advocate-Auftrag, prüft eine fertige Optionsanalyse gegen vier
  Kategorien (faktische Stimmigkeit gegen Quellen, methodische Lücken,
  sprachliche Schmuggelware, Quellenverwendung/Tier-Disziplin). Bewusst
  anderes Modell als das produzierende: Default `gemma3:27b` (anderer Bias als
  Qwen, sequentielles Modell-Swapping über Ollama-Eviction). Modell ist über
  env-Var `CRITIQUE_MODEL` umstellbar. Wird über `query.py --critique`
  zugeschaltet (default off, ~30 % Latenz-Aufschlag).
  Implementierungs-Caveats: `num_ctx=8192` (Default 4096 reicht nicht für
  Frage+Chunks+Analyse+Prüfraster und führt mit langchain-ollama-Streaming
  zu Runner-Crashes), `reasoning=False` (sonst landet bei thinking-fähigen
  Modellen die Antwort manchmal komplett im `thinking`-Feld → leeres
  `response`).
- `scripts/query.py` — LCEL-Pipeline mit Tier-aware Retrieval per Default.
  Holt `fetch_k=20` Kandidaten per Vektor-Similarity, gewichtet mit Tier-Boost,
  liefert die top-5 an die LLM. Bei deutlich besser passenden Tier-2/3-Chunks
  gewinnen diese trotzdem — Soft-Preference, kein Hard-Filter. CLI-Flag
  `--no-tier-boost` für A/B-Vergleich. Pipeline-Flags:
  `--structure` (Stage 4 JSON), `--verify` (Stage 3 Claim-Verifikation,
  setzt verification_status + evidence_quote), `--critique` (Stage 5),
  `--notion` (Stage 6 Push). `--verify` und `--notion` implizieren
  beide `--structure`. Ausgabe der Quellen enthält Tier, Label,
  Similarity und kombinierten Score zur Transparenz.
- `scripts/generate_post.py` — gleiche Tier-aware Pipeline wie `query.py`,
  ebenfalls mit `--no-tier-boost`. Plattform-Prompts sind auf
  Optionsanalyse umgestellt: für Twitter/Instagram bewusst nur
  Faktenlage + Trade-off/Wertfrage benennen (ohne Position), für
  LinkedIn die volle 6-Stufen-Struktur. Bei Twitter/Instagram steht
  explizit "KEINE Position, KEINE Empfehlung" im Prompt. Plattform-
  Prompts und Modelfile-SYSTEM bilden zusammen die methodische Klammer.

`langchain-classic` ist aus `requirements.in` entfernt (nur noch transitiv in
`requirements.txt`, verschwindet beim nächsten `pip-compile`-Lauf).

- `config/source_tiers.yaml` — Quellen-Hierarchie (Tier 1 = Primärquellen
  Recht/Statistik/Parlament, Tier 2 = Qualitätsmedien regional und
  überregional, Tier 3 = Default für nicht klassifizierte Quellen). Wird
  von `ingest.py` beim Start geladen. Pflege erfolgt hier, nicht im Code.
- `scripts/schema.py` — Pydantic-Modelle für Optionsanalyse, Faktum, Norm,
  Option. Faktum hat `verifiziert: bool` (heuristisch), `verification_status`
  (Stage-3-Output, vier-stufig) und `evidence_quote`. `render_optionsanalyse`
  rendert ins Markdown mit Status-Markern (✅/🟡/⚪/❌).
- `scripts/structure.py` — Stage 4 Post-Processing: Markdown → Pydantic-JSON
  via separatem LLM-Aufruf mit `format="json"` und Pydantic-Validierung
  (ein Retry bei Schema-Fehler). Default-Modell `gemma3:27b`, env-Var
  `STRUCTURE_MODEL`.
- `scripts/verify.py` — Stage 3: jede Aussage in `faktenlage` einzeln
  gegen Tier-1-Chunks prüfen, `verification_status` und `evidence_quote`
  setzen. Batched: alle Claims in einem LLM-Aufruf, Ergebnis als
  `{verifications: [...]}`-Wrapper, Code unwrappt. Default `gemma3:27b`,
  env-Var `VERIFY_MODEL`.
- `scripts/notion_sink.py` — Stage 6: Notion-REST-API, Block-Renderer für
  alle 6 Sektionen, Toggle-Block für Critique, evidence_quote als nested
  Quote-Block unter dem zugehörigen Faktum. Env-Vars `NOTION_TOKEN` und
  `ATHENA_NOTION_PARENT` (Default = ID der "🧭 Optionsanalysen"-Subpage
  unter "🏛️ Projekt Athena – Übersicht").

**Remote-Setup-Stand (auf `aitest`/100.101.225.56):**
- Ollama-Service hört auf allen Interfaces (`OLLAMA_HOST=0.0.0.0:11434` als
  systemd-Override, parallel zum bestehenden `HSA_OVERRIDE_GFX_VERSION=11.0.2`).
- Modelle vorhanden: `athena:latest` (gebaut aus dem Modelfile),
  `qwen3.6:35b-a3b` (Base), `nomic-embed-text` (Embeddings). Daneben
  fremde Workloads (`alfred:latest` u.a.) — Box ist multi-tenant.
- Smoke-Test über HTTP-API erfolgreich, Pfofeld-Persona kommt korrekt durch.

## Arbeitsweise

- Vor strukturellen Änderungen (neue Module, neue Dependencies, Verzeichnis-
  umbau) einen Plan vorlegen und Bestätigung einholen.
- Code-Kommentare und User-facing Strings auf Deutsch, Funktions- und
  Variablennamen englisch (wie bisher).
- Ab Pipeline-Stufe 3 müssen pro Stufe Tests existieren.
- Quellen niemals verbatim zitieren — paraphrasieren mit Quellenangabe und Tier.
- Bei methodischen Unsicherheiten nachfragen, nicht raten. Das Projekt steht
  und fällt mit methodischer Disziplin.