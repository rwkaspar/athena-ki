# Themen-Dossier-Schema v1

Jedes Dossier ist eine JSON-Datei `<slug>.json`. Build-Funktion `build_themen_dossiers()`
in `evidenz-partei/build.py` rendert sie als `dist/themen/<slug>.html`.

## Struktur

```json
{
  "slug": "migration",
  "title": "Migration & Asyl",
  "lead": "Ein Satz, der das Themenfeld einordnet.",
  "evidenz_position_url": "programm/migration-asyl.html",
  "last_reviewed": "2026-06-09",
  "facts_at_a_glance": [
    {
      "label": "Was wird gemessen",
      "value": "konkrete Zahl",
      "year": "2024",
      "source": "Herausgeber",
      "source_url": "https://..."
    }
  ],
  "legal_frame": [
    {
      "law": "Art. 16a GG",
      "summary": "1-2 Sätze, was die Norm regelt.",
      "source_url": "https://..."
    }
  ],
  "party_positions": [
    {
      "party": "SPD",
      "position_summary": "Worauf zielt die Position? 2-3 Sätze.",
      "key_claim": "Die zentrale Sachbehauptung der Position.",
      "claim_source": "Wahlprogramm 2025 / BT-Rede / etc.",
      "claim_source_url": "https://...",
      "athena_analysis": {
        "status": "vorab|in-arbeit|verifiziert",
        "fact_check": "Stimmt die zentrale Behauptung empirisch? Mit Tier-1-Belegen.",
        "trade_offs": "Was passiert, wenn die Position umgesetzt wird? Welche Werte werden gestärkt, welche kosten sie?",
        "value_assumptions": "Welche Wertentscheidungen liegen der Position zugrunde (nicht-empirisch)?",
        "verdict": "gestützt|teilweise gestützt|widerlegt|wertfundamental|datenmäßig offen",
        "verdict_reasoning": "1-3 Sätze: warum dieses Verdikt?",
        "tier1_sources": [
          {"label": "Quelle", "url": "https://..."}
        ]
      }
    }
  ],
  "open_questions": [
    "Streitfragen, bei denen die Empirie dünn ist und Werteentscheidungen dominieren."
  ]
}
```

## Verdikt-Definitionen

- **gestützt**: zentrale empirische Behauptung wird durch Tier-1-Quellen klar bestätigt.
- **teilweise gestützt**: Kernbehauptung trifft mit erheblichen Einschränkungen zu.
- **widerlegt**: zentrale empirische Behauptung wird durch Tier-1-Quellen klar widerlegt.
- **wertfundamental**: Position lässt sich nicht empirisch prüfen, sondern beruht auf Wertentscheidung. Athena macht die Wertannahme transparent statt zu bewerten.
- **datenmäßig offen**: Faktenlage ist dünn, methodisch umstritten oder fehlt.

## Status-Lebenszyklus

- **vorab**: Erste Skizze durch Claude (Wissensstand 01/2026). Nicht durch Athena-Pipeline.
- **in-arbeit**: Athena-Lauf läuft oder wartet auf Critique-Pass.
- **verifiziert**: durch alle drei Stufen gelaufen (Proposer → Critique → Adversarial Verify), Tier-1-Quellen sind aus dem RAG, nicht aus dem LLM-Wissen.

Bis aitest + Critique-Pass-Setup produktiv sind, stehen die meisten Athena-Analysen
auf `vorab`. Die Seite zeigt dann einen klaren Hinweis-Sticker.
