"""Athena - Stage 6 Documentation Sink: Notion-API.

Schiebt eine Optionsanalyse (Pydantic) plus optional Critique und Quellenliste
als neue Page in den "🧭 Optionsanalysen"-Subtree der "Projekt Athena"-Notion-
Seite. Default-Parent ist die zentrale Optionsanalysen-Page; via env-Var
`ATHENA_NOTION_PARENT` umstellbar (z. B. wenn du eine eigene Database mit
filterbaren Properties anlegst, dann diese Data-Source-ID als Parent setzen).

Voraussetzung: env-Var `NOTION_TOKEN` mit Internal Integration Token. Die
Integration muss Schreibzugriff auf den Parent haben — einmalig manuell:
Notion → "🧭 Optionsanalysen" öffnen → Connect → Athena-RAG hinzufügen.
"""

import os
from datetime import datetime, timezone

import requests

from schema import Optionsanalyse

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
DEFAULT_PARENT = "35832d54-09c7-8116-8d0a-da091c53fb5d"  # 🧭 Optionsanalysen
ATHENA_NOTION_PARENT = os.getenv("ATHENA_NOTION_PARENT", DEFAULT_PARENT)

TIER_LABEL = {1: "Tier 1 (Primär)", 2: "Tier 2 (Medien)", 3: "Tier 3 (Kommentar)"}
KONFIDENZ_EMOJI = {"hoch": "🟢", "mittel": "🟡", "niedrig": "🔴"}
STATUS_MARKER = {
    "verifiziert": "✅",
    "teilweise": "🟡",
    "nicht_belegt": "⚪",
    "widersprochen": "❌",
}


def _rt(text: str) -> list[dict]:
    """Rich-text-Array für einen Plain-String. Notion limitiert eine
    rich-text-Range auf 2000 Zeichen — bei längeren Strings splitten."""
    LIMIT = 2000
    if len(text) <= LIMIT:
        return [{"type": "text", "text": {"content": text}}]
    chunks = [text[i:i + LIMIT] for i in range(0, len(text), LIMIT)]
    return [{"type": "text", "text": {"content": c}} for c in chunks]


def _heading(level: int, text: str) -> dict:
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": _rt(text)}}


def _paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(text)}}


def _bullet(text: str, children: list[dict] | None = None) -> dict:
    block = {"object": "block", "type": "bulleted_list_item",
             "bulleted_list_item": {"rich_text": _rt(text)}}
    if children:
        block["bulleted_list_item"]["children"] = children
    return block


def _quote(text: str) -> dict:
    return {"object": "block", "type": "quote", "quote": {"rich_text": _rt(text)}}


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _callout(text: str, emoji: str = "ℹ️") -> dict:
    return {"object": "block", "type": "callout", "callout": {
        "rich_text": _rt(text),
        "icon": {"type": "emoji", "emoji": emoji},
    }}


def _toggle(summary: str, children: list[dict]) -> dict:
    return {"object": "block", "type": "toggle", "toggle": {
        "rich_text": _rt(summary),
        "children": children,
    }}


def _build_blocks(
    question: str,
    analysis: Optionsanalyse,
    sources: list[str],
    critique: str | None,
) -> list[dict]:
    blocks: list[dict] = []

    blocks.append(_heading(1, question))

    konf = KONFIDENZ_EMOJI.get(analysis.konfidenz, "⚪")
    meta = (
        f"Datum: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
        f"Frage-Typ: {analysis.frage_typ}  ·  "
        f"Konfidenz: {konf} {analysis.konfidenz}"
    )
    blocks.append(_callout(meta, "📐"))

    blocks.append(_heading(2, "Faktenlage"))
    if analysis.faktenlage:
        for f in analysis.faktenlage:
            marker = STATUS_MARKER.get(
                f.verification_status,
                "✓" if f.verifiziert else "○",  # Fallback wenn Stage 3 nicht lief
            )
            tier_str = TIER_LABEL.get(f.tier, "Trainingswissen") if f.tier else "Trainingswissen"
            chunk_str = f", chunk {f.quelle_chunk}" if f.quelle_chunk is not None else ""
            children = [_quote(f.evidence_quote)] if f.evidence_quote else None
            blocks.append(_bullet(f"{marker} {f.aussage}  ({tier_str}{chunk_str})", children=children))
    else:
        blocks.append(_paragraph("(keine belastbaren Fakten)"))

    if analysis.rechtsrahmen:
        blocks.append(_heading(2, "Rechtsrahmen"))
        for n in analysis.rechtsrahmen:
            chunk_str = f"  [chunk {n.quelle_chunk}]" if n.quelle_chunk is not None else ""
            blocks.append(_bullet(f"{n.bezeichnung}{chunk_str} — {n.relevanz}"))

    if analysis.optionen:
        blocks.append(_heading(2, "Optionen mit Trade-offs"))
        for i, opt in enumerate(analysis.optionen, 1):
            blocks.append(_heading(3, f"Option {chr(64+i)}: {opt.titel}"))
            blocks.append(_paragraph(opt.beschreibung))
            blocks.append(_paragraph("Trade-offs:"))
            for t in opt.trade_offs:
                blocks.append(_bullet(t))
            blocks.append(_paragraph("Wertannahmen:"))
            for w in opt.wertannahmen:
                blocks.append(_bullet(w))

    if analysis.vergleichsfaelle:
        blocks.append(_heading(2, "Vergleichsfälle"))
        for v in analysis.vergleichsfaelle:
            blocks.append(_bullet(v))

    if analysis.offene_fragen:
        blocks.append(_heading(2, "Offene Fragen"))
        for q in analysis.offene_fragen:
            blocks.append(_bullet(q))

    if sources:
        blocks.append(_heading(2, "Quellen"))
        for s in sources:
            blocks.append(_bullet(s))

    if critique:
        blocks.append(_divider())
        critique_blocks = [_paragraph(p) for p in critique.split("\n\n") if p.strip()]
        blocks.append(_toggle("🔍 Critique-Pass (Devil's Advocate, anderes Modell)", critique_blocks))

    return blocks


def _truncate_title(question: str, limit: int = 70) -> str:
    q = question.strip().rstrip("?")
    if len(q) <= limit:
        return q + "?"
    return q[:limit].rsplit(" ", 1)[0] + "…"


def publish_to_notion(
    question: str,
    analysis: Optionsanalyse,
    sources: list[str],
    critique: str | None = None,
    parent_id: str | None = None,
) -> str:
    """Erstellt eine Notion-Page unter dem konfigurierten Parent. Gibt URL zurück."""
    if not NOTION_TOKEN:
        raise RuntimeError(
            "NOTION_TOKEN env-Var nicht gesetzt. Setup: Notion → Settings → "
            "Connections → Develop your own integrations → Internal Integration "
            "Token erstellen → 'export NOTION_TOKEN=...'. Dann Athena-RAG-"
            "Integration mit der Optionsanalysen-Seite verbinden."
        )

    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    parent = parent_id or ATHENA_NOTION_PARENT
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    title = f"{_truncate_title(question)} · {timestamp}"

    payload = {
        "parent": {"page_id": parent},
        "icon": {"type": "emoji", "emoji": "🧭"},
        "properties": {
            "title": [{"type": "text", "text": {"content": title}}],
        },
        "children": _build_blocks(question, analysis, sources, critique),
    }

    r = requests.post(f"{NOTION_API}/pages", headers=headers, json=payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Notion-API-Fehler {r.status_code}: {r.text}")
    return r.json().get("url", "(URL nicht zurückgegeben)")
