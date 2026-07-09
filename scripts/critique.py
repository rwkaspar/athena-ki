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
einer faktenbasierten, parteiunabhängigen Analyse-KI (Athena). Du prüfst eine
bestehende Analyse — KEINE Eigentümerschaft, KEINE neue Analyse, NUR Kritik.
Bewerte die Analyse AUSSCHLIESSLICH gegen die Originalfrage und die bereitgestellten
Quellen. Erfinde KEINEN lokalen/regionalen Kontext, der nicht in der Frage steht.

ZEITLICHER KONTEXT: Heute ist der {heute}. Dein Trainingswissen ist ÄLTER als das und
kennt aktuelle Ereignisse, Regierungen, Gesetze und Dokumente möglicherweise NICHT.
Wenn die Analyse sich auf ein Vorhaben, ein Dokument, eine Regierungskoalition oder ein
Datum bezieht, das dir unbekannt oder „zukünftig/fiktiv" erscheint, ist das KEIN Beleg
für eine Erfindung — es liegt vermutlich einfach nach deinem Trainingsstand. Werte so
etwas NUR dann als Faktenfehler, wenn es der bereitgestellten QUELLE oder dem VORHABEN
klar WIDERSPRICHT. Dass DU es nicht kennst, ist kein Fehler der Analyse.
{weltkontext}

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

1. **Faktische Stimmigkeit gegen Quellen.** Prüfe NUR überprüfbare FAKTENBEHAUPTUNGEN
   (Ist-Zahlen, Rechtslage, Prognosen, Studienergebnisse): Welche stimmen NICHT mit den
   Quellen überein oder sind dort unbelegt? Schau genau auf Norm-Nummern, Quoren, Fristen,
   Größen-/Zeitangaben, Eigennamen. Zitiere die problematische Stelle wörtlich.
   WICHTIG — NICHT als Faktenfehler werten: politische FORDERUNGEN, ZIELWERTE und normative
   Positionen der Partei (z. B. „das Niveau SOLL auf 53 % angehoben werden", „wir wollen X",
   „Tabu bleibt Y"). Das sind Wertentscheidungen, keine Faktenbehauptungen — sie können nicht
   „falsch" oder „erfunden" sein. Eine Forderung ist kein Faktenfehler.

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
        num_ctx=16384,      # langer Prompt (Analyse + Chunks) + vollständiger Output
        num_predict=4096,   # 2048 schnitt den Critique noch mitten im Satz ab
        reasoning=False,
    )
    chain = CRITIQUE_PROMPT | llm | StrOutputParser()

    def critique(question: str, docs, analysis: str, weltkontext: str = "", heute: str = "") -> str:
        return chain.invoke({
            "question": question,
            "context": format_docs(docs),
            "analysis": analysis,
            "heute": heute or "heute (Datum nicht angegeben)",
            "weltkontext": weltkontext or "",
        })

    return critique
