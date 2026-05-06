"""Athena - Critique-Pass: zweites LLM prüft eine bestehende Optionsanalyse
auf methodische und faktische Schwächen.

Designprinzip: Das Critique-Modell hat einen anderen Bias als das
produzierende Modell (Default: Gemma 3 statt Qwen 3.6) und sieht *nur*
die fertige Analyse plus die Originalquellen — es macht keine eigene
Analyse, sondern reviewt. Das ist Stage 5 der 6-stufigen Pipeline aus
claude.md.
"""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from retrieval import format_docs

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CRITIQUE_MODEL = os.getenv("CRITIQUE_MODEL", "gemma3:27b")

CRITIQUE_PROMPT = PromptTemplate.from_template(
    """Du bist ein methodischer Reviewer für strukturierte politische Optionsanalysen
einer KI-gestützten Bürgermeisterin (Athena, Gemeinde Pfofeld). Du prüfst eine
bestehende Analyse — KEINE Eigentümerschaft, KEINE neue Analyse, NUR Kritik.

Deine Aufgabe ist es, konkrete Schwächen der Analyse zu finden — nicht die Analyse
zu wiederholen oder zu verbessern. Antworte strukturiert in vier Kategorien.
Wenn eine Kategorie keine Beanstandungen hat, schreib das explizit
("Keine Beanstandungen.").

Originalfrage:
{question}

Bereitgestellte Quellen-Chunks (mit Quelle und Tier-Hinweis):
{context}

Zu prüfende Analyse:
{analysis}

Prüfraster:

1. **Faktische Stimmigkeit gegen Quellen.** Welche Behauptungen in der Analyse
   stimmen NICHT mit den bereitgestellten Quellen überein? Schau besonders genau
   auf: Norm-Nummern (Artikel, Paragraphen), Quoren, Fristen, Größen- und
   Zeitangaben, Eigennamen. Zitiere die problematische Stelle wörtlich.

2. **Methodische Lücken.** Welche relevanten Lösungsoptionen wurden nicht erwogen?
   Welche Wertannahmen fehlen oder sind nicht expliziert? Welche Vergleichsfälle
   wären naheliegend gewesen, kommen aber nicht vor?

3. **Sprachliche Schmuggelware.** Enthält die Analyse trotz Optionsanalyse-Auftrag
   implizite Empfehlungen? Suche nach Formulierungen wie "vieles spricht für ...",
   "der naheliegende Schritt ist ...", "empfehlenswert wäre ...", "in der Praxis
   bewährt sich ...". Zitiere wörtlich.

4. **Quellenverwendung und Tier-Disziplin.** Werden Aussagen, die NICHT aus den
   bereitgestellten Quellen stammen, sauber als Trainingswissen oder Annahme
   gekennzeichnet? Werden Tier-1-Primärquellen bevorzugt zitiert? Werden
   Tier-2/3-Quellen entsprechend transparent gemacht?

Sei konkret. Vermeide allgemeine Bemerkungen wie "die Analyse könnte vertieft
werden" — solche Sätze haben keinen Wert. Wenn du nichts Konkretes findest,
schreib "Keine Beanstandungen.".
"""
)


def create_critique_chain(model: str | None = None, host: str | None = None):
    """Critique-Chain bauen. Rückgabe ist eine Funktion
    critique(question, docs, analysis) -> str.

    num_ctx wird explizit auf 8192 gesetzt: Critique-Prompts sind oft lang
    (Originalfrage + Chunks + komplette Analyse + Prüfraster), und Ollamas
    Default von 4096 Tokens führt mit langchain-ollama-Streaming zu
    Runner-Crashes statt sauberer Truncation."""
    llm = OllamaLLM(
        model=model or CRITIQUE_MODEL,
        base_url=host or OLLAMA_HOST,
        timeout=900,
        num_ctx=8192,
        reasoning=False,
    )
    chain = CRITIQUE_PROMPT | llm | StrOutputParser()

    def critique(question: str, docs, analysis: str) -> str:
        return chain.invoke({
            "question": question,
            "context": format_docs(docs),
            "analysis": analysis,
        })

    return critique
