"""Athena - Stage 4 Post-Processing: Markdown → strukturiertes Pydantic-JSON.

Athenas Optionsanalyse kommt aus query.py als Markdown. Für Downstream-Stages
(Notion-Sink, Fine-Tuning-Daten, Field-targeted Critique) brauchen wir
maschinenlesbares JSON nach `schema.Optionsanalyse`. Statt das produzierende
Modell mit strict-JSON zu belasten (was Ollama 0.18.3 nicht zuverlässig
erzwingt), macht ein separater Strukturierungs-Aufruf die Konvertierung —
einfache Aufgabe, `format="json"` reicht, Pydantic validiert, ein Retry bei
Schema-Verletzung.

Default-Modell: gemma3:27b (gleiches wie Critique, eh schon im Cache, gut beim
Strukturieren). Über env-Var `STRUCTURE_MODEL` umstellbar.
"""

import json
import os

from langchain_ollama import OllamaLLM
from pydantic import ValidationError

from schema import Optionsanalyse

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
STRUCTURE_MODEL = os.getenv("STRUCTURE_MODEL", "gemma3:27b")

STRUCTURE_PROMPT = """Du konvertierst eine Athena-Optionsanalyse aus Markdown in striktes JSON.
Halte dich EXAKT an die unten beschriebenen Felder. Erfinde keine Inhalte, die nicht im Markdown stehen — wenn ein Feld nicht abgedeckt ist, lass es leer ([] oder null).

Schema-Felder:
- thema (string): Konkretes Thema der Frage in einem Satz
- frage_typ (string, einer von "wissensfrage" oder "entscheidungsfrage"): "wissensfrage" wenn die Antwort reine Auskunft ist (keine Optionen), "entscheidungsfrage" wenn normativer Spielraum besteht
- faktenlage (array von Objekten): jedes Objekt hat:
    - aussage (string): paraphrasierte Faktaussage
    - quelle_chunk (integer oder null): Index des Chunks falls explizit referenziert (z. B. "[chunk 2]"), sonst null
    - tier (integer 1, 2, 3, oder null): Tier der Quelle falls erkennbar, sonst null
    - verifiziert (boolean): true wenn Quelle Tier 1 und Aussage daraus stammt, sonst false
- rechtsrahmen (array von Objekten): jedes Objekt hat:
    - bezeichnung (string): z. B. "Art. 18 Abs. 2 BayGO"
    - relevanz (string): warum die Norm einschlägig ist
    - quelle_chunk (integer oder null)
- optionen (array von Objekten, NUR wenn frage_typ="entscheidungsfrage", 2-4 Stück): jedes Objekt hat:
    - titel (string)
    - beschreibung (string)
    - trade_offs (array von string)
    - wertannahmen (array von string)
- vergleichsfaelle (array von string)
- offene_fragen (array von string)
- konfidenz (string, einer von "hoch", "mittel", "niedrig")

Die Analyse:
---
{markdown}
---

Antworte mit reinem JSON-Objekt, ohne Markdown-Codefences, ohne erklärenden Text vorher oder nachher."""


def _build_llm(model: str | None, host: str | None):
    return OllamaLLM(
        model=model or STRUCTURE_MODEL,
        base_url=host or OLLAMA_HOST,
        timeout=600,
        reasoning=False,
        format="json",
    )


def structure_analysis(
    markdown: str,
    model: str | None = None,
    host: str | None = None,
) -> Optionsanalyse:
    """Markdown-Analyse in Optionsanalyse parsen. Bei Schema-Validierungsfehler
    EIN Retry mit der Fehlermeldung als zusätzlichem Hinweis."""
    llm = _build_llm(model, host)
    prompt = STRUCTURE_PROMPT.format(markdown=markdown)

    raw = llm.invoke(prompt)
    try:
        return Optionsanalyse.model_validate_json(raw)
    except ValidationError as e:
        retry_prompt = (
            prompt
            + "\n\nDer vorherige Versuch war ungültig: "
            + str(e)
            + "\n\nKorrigiere die Struktur und antworte erneut nur mit JSON."
        )
        raw_retry = llm.invoke(retry_prompt)
        return Optionsanalyse.model_validate_json(raw_retry)
