"""Athena - Stage 3: Claim Extraction & Verification.

Prüft jede atomare Aussage in `analysis.faktenlage` einzeln gegen die
bereitgestellten Quellen-Chunks. Macht das `verifiziert`-Feld erstmals
belastbar und liefert ein wörtliches Quote aus der Quelle.

Anders als Stage 5 (Critique) ist das hier reine Faktentreue, kein
Methoden-Review. Wir senden alle Claims in einem Batch-Prompt an ein
LLM mit `format="json"`, kriegen ein strukturiertes Verifikations-Array
zurück und mergen es in die bestehende Optionsanalyse.

Default-Modell: gemma3:27b (gleiches wie Critique/Structure — gut beim
Vergleichen von Aussagen mit Belegtext, kein Reasoning-Modell). Über
env-Var `VERIFY_MODEL` umstellbar.
"""

import json
import os

from langchain_ollama import OllamaLLM

from schema import Faktum, Optionsanalyse

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
VERIFY_MODEL = os.getenv("VERIFY_MODEL", "gemma3:27b")

VERIFY_PROMPT = """Du verifizierst Faktaussagen aus einer politischen Optionsanalyse gegen die ursprünglichen Quellen-Chunks. Reine Faktentreue, kein Methoden-Review, keine eigenen Analysen.

Bewertungs-Schema pro Aussage:
- "verifiziert": Tier-1-Primärquelle (Recht, amtliche Statistik, Parlamentsdokument) in den Chunks belegt die Aussage wörtlich oder paraphrasierend.
- "teilweise": Eine Quelle bestätigt einen Kernbestandteil, aber nicht alle Aspekte; oder nur Tier-2/3-Quelle deckt es ab.
- "nicht_belegt": Keiner der bereitgestellten Chunks deckt die Aussage. Sie kann trotzdem korrekt sein, ist aber aus den Quellen heraus nicht prüfbar.
- "widersprochen": Ein Chunk sagt explizit etwas anderes.

Aussagen zu prüfen:
{claims}

Quellen-Chunks:
{chunks}

Antworte als JSON-Objekt mit dem Schlüssel `verifications`, der ein Array enthält — ein Eintrag pro Aussage, in derselben Reihenfolge wie oben. Kein Markdown-Codefence.

{{
  "verifications": [
    {{
      "claim_idx": 0,
      "status": "verifiziert" | "teilweise" | "nicht_belegt" | "widersprochen",
      "evidence_chunk": <int oder null>,
      "evidence_quote": "<wörtliches Zitat aus dem Chunk, oder null wenn nicht_belegt>"
    }},
    {{ "claim_idx": 1, ... }},
    ...
  ]
}}
"""


_VALID_STATUS = {"verifiziert", "teilweise", "nicht_belegt", "widersprochen"}


def _format_claims(faktenlage: list[Faktum]) -> str:
    return "\n".join(f"{i}. {f.aussage}" for i, f in enumerate(faktenlage))


def _format_chunks(docs) -> str:
    blocks = []
    for i, d in enumerate(docs):
        tier = d.metadata.get("tier_rank", "?")
        blocks.append(f"[CHUNK {i}] (Tier {tier})\n{d.page_content}")
    return "\n\n".join(blocks)


def verify_claims(
    analysis: Optionsanalyse,
    docs,
    model: str | None = None,
    host: str | None = None,
) -> Optionsanalyse:
    """Verifiziert jede Aussage in analysis.faktenlage gegen docs. Mutiert
    und gibt die Optionsanalyse zurück. Bei leerem faktenlage no-op."""
    if not analysis.faktenlage:
        return analysis

    llm = OllamaLLM(
        model=model or VERIFY_MODEL,
        base_url=host or OLLAMA_HOST,
        timeout=600,
        reasoning=False,
        format="json",
    )
    prompt = VERIFY_PROMPT.format(
        claims=_format_claims(analysis.faktenlage),
        chunks=_format_chunks(docs),
    )

    raw = llm.invoke(prompt)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Stage 3 darf nicht die ganze Pipeline killen — bei JSON-Fehler
        # bleibt die Analyse unverändert (Stage-4-Heuristik bleibt drin).
        print(f"   ⚠️  Verify: JSON-Parse fehlgeschlagen, Stage 3 übersprungen")
        return analysis

    # Manche Modelle wrappen das Array in einem Objekt {"verifications": [...]}
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break

    if not isinstance(data, list):
        print(f"   ⚠️  Verify: unerwartetes JSON-Format, Stage 3 übersprungen")
        return analysis

    by_idx = {item.get("claim_idx"): item for item in data if isinstance(item, dict)}

    for i, faktum in enumerate(analysis.faktenlage):
        item = by_idx.get(i)
        if not item:
            continue
        status = item.get("status")
        if status in _VALID_STATUS:
            faktum.verification_status = status
            faktum.verifiziert = (status == "verifiziert")
        evidence_chunk = item.get("evidence_chunk")
        if isinstance(evidence_chunk, int):
            faktum.quelle_chunk = evidence_chunk
        evidence_quote = item.get("evidence_quote")
        if isinstance(evidence_quote, str) and evidence_quote.strip():
            faktum.evidence_quote = evidence_quote.strip()

    return analysis
