# Athena-Benchmark v1 — Faktenwissen Bundespolitik

Antwort auf die Frage „Ist Athena für diesen Use Case geeignet?" — falsifizierbar
und reproduzierbar. Misst Athena (RAG + Tier-klassifizierte Quellen) gegen
Standard-LLMs (ChatGPT/Claude/plain Mistral, ohne RAG) auf drei Achsen:

1. **Faktentreue**: Trifft die Antwort den Wert aus der Tier-1-Primärquelle?
2. **Halluzinations-Rate**: Wie viele zitierte URLs/Paragraphen sind erfunden?
3. **Quellen-Tier-Verteilung**: Anteil Primärquellen (Tier 1) in den Belegen.

## Dateien

- `gold_set_v1.json` — 30 Fragen mit verifizierter Soll-Antwort + accept_patterns
  (Regex für Auto-Grading) + hallucination_traps (typische Fehlantworten).
- `runs/` — Ergebnis-Dumps pro Lauf (`<datum>_<modell>.jsonl`), nicht im Repo.
- `reports/` — Zusammenfassungen pro Lauf (`<datum>_report.md`).

## Workflow

### 1. Gold-Set verifizieren (einmalig)

Jede Frage mit `verify_needed: true` durchgehen:

1. Die `gold_source`-URL öffnen.
2. Den `gold_answer` gegen den Quelltext prüfen.
3. Bei Abweichung: `gold_answer` + `accept_patterns` korrigieren.
4. `verify_needed` auf `false` setzen.

Begründung: ein Benchmark, der Halluzinationen messen soll, darf selbst keine
enthalten. Die mit `verify_needed: true` markierten Posten sind Vorschläge
ohne Quellenprüfung (Wissens-Cutoff Claude: Jan 2026 — manche Werte
2024/2025/2026 brauchen Live-Verifikation).

### 2. Benchmark-Lauf

Drei Modi für Vergleichbarkeit:

| Modus | Modell | RAG | Zweck |
|---|---|---|---|
| `athena` | mistral-large (über serve.py) | ja | Unser System |
| `bare-mistral` | mistral-large (API direkt) | nein | Gleiches Modell ohne RAG |
| `bare-claude` | claude-opus-4-x | nein | Anderes Familienzusammensetzung |
| `bare-chatgpt` | gpt-5 oder gpt-4o | nein | Marktüblicher Vergleich |

Lauf-Skript folgt in `scripts/run_gold_eval.py` (TBD). Bis dahin per Hand:
für jede Frage die Antwort des Modells erfassen, in JSONL ablegen.

### 3. Auswertung

Für jede Antwort gegen das Gold-Set:

- **Hit**: ≥1 `accept_patterns` matcht (oder `min_pattern_matches` Patterns
  bei Multi-Fakt-Fragen wie der AI-Act-Liste).
- **Halluzinations-Score**: jede im Antworttext zitierte URL gegen reale
  Existenz prüfen (HEAD-Request 200/3xx ja, sonst nein) + jeder Paragraf
  gegen gesetze-im-internet.de.
- **Tier-1-Anteil**: zitierte Domains gegen `config/source_tiers_bund.yaml`
  matchen, Tier-Verteilung berechnen.

### 4. Report

Markdown mit:
- Hit-Rate je Modus (Athena vs. Baselines)
- Halluzinations-Rate je Modus
- Tier-Verteilung je Modus
- Liste der Fragen, die nur Athena richtig hat (Hauptargument)
- Liste der Fragen, die kein Modus richtig hat (gemeinsame Lücke)

## Themenabdeckung Gold-Set v1

Schuldenbremse (3), Bürgergeld (2), Rente (2), Steuersystem (2), KI-Regulation
(3), Klima (2), Energie (1), Migration (2), Verteidigung (2), Wahlrecht (2),
Parlamentarismus (2), Parteienfinanzierung (2), Bildung (2), Gesundheit (1),
Außenpolitik (1), Direkte Demokratie (1) = 30 Fragen.

## Erweiterung v2 (Roadmap)

- Multi-Step-Fragen („Welche Optionen gäbe es zur Reform X?") — bewertet
  Optionsanalyse-Qualität statt nur Fakten.
- Trap-Set: Fragen, bei denen die korrekte Antwort „dazu gibt es keine
  belastbaren Daten" lautet. Misst die Bescheidenheit des Modells.
- Drei-Modell-Adversarial-Verify integriert: nach jedem Modell-Antwort durch
  Qwen3.6-MoE + Gemma4-31B als Devil's Advocate prüfen.
