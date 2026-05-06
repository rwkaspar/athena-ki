"""Athena - Strukturschema für Optionsanalysen (Stage 4 der Pipeline).

Pydantic-Modelle, die das methodische Kernprinzip aus claude.md in eine
maschinenlesbare Form gießen. Das LLM produziert JSON nach diesem Schema
(via Ollamas format-Parameter), Critique-Pass kann Felder referenzieren,
Notion-Sink kann jedes Feld in einen passenden Block rendern.

Faktenfragen vs. Entscheidungsfragen werden über `frage_typ` getrennt:
faktisch → optionen/wertannahmen/vergleichsfaelle bleiben leer.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Faktum(BaseModel):
    """Ein einzelner Claim mit Quellennachweis und Verifikationsstatus."""
    aussage: str = Field(..., description="Die Faktaussage selbst, paraphrasiert")
    quelle_chunk: int | None = Field(
        None,
        description="Index des Chunks in den bereitgestellten Quellen (0-basiert), wenn die Aussage daraus stammt; None bei Trainingswissen",
    )
    tier: Literal[1, 2, 3] | None = Field(
        None,
        description="Tier der Quelle (1=Primär, 2=Medien, 3=Kommentar); None wenn Trainingswissen",
    )
    verifiziert: bool = Field(
        ...,
        description="True nur wenn die Aussage durch eine Tier-1-Primärquelle in den Chunks belegt ist (heuristisch aus Stage 4; durch Stage-3-Verifikation überschrieben)",
    )
    verification_status: Literal["verifiziert", "teilweise", "nicht_belegt", "widersprochen"] | None = Field(
        None,
        description="Stage-3-Verifikationsergebnis. None solange Stage 3 nicht gelaufen ist.",
    )
    evidence_quote: str | None = Field(
        None,
        description="Wörtliches Zitat aus dem belegenden Chunk (Stage-3-Output)",
    )


class Norm(BaseModel):
    """Eine Rechtsnorm im Rechtsrahmen."""
    bezeichnung: str = Field(..., description="Norm-Kennzeichnung, z. B. 'Art. 18 Abs. 2 BayGO'")
    relevanz: str = Field(..., description="Warum diese Norm für die Frage einschlägig ist")
    quelle_chunk: int | None = Field(None, description="Chunk-Index, falls in den Quellen wörtlich genannt")


class Option(BaseModel):
    """Eine Lösungsoption mit Trade-offs und Wertannahmen."""
    titel: str = Field(..., description="Kurzer prägnanter Titel der Option")
    beschreibung: str = Field(..., description="Worum geht es konkret bei dieser Option?")
    trade_offs: list[str] = Field(
        ...,
        min_length=1,
        description="Vor- UND Nachteile dieser Option, jeder Punkt einzeln",
    )
    wertannahmen: list[str] = Field(
        ...,
        min_length=1,
        description="Welche Werte oder Prioritäten setzt diese Option implizit voraus?",
    )


class Optionsanalyse(BaseModel):
    """Vollständige Athena-Optionsanalyse (Stage 4 der Pipeline)."""
    thema: str = Field(..., description="Konkretes Thema der Frage in einem Satz")
    frage_typ: Literal["wissensfrage", "entscheidungsfrage"] = Field(
        ...,
        description="wissensfrage = rein faktische Auskunft; entscheidungsfrage = normativer Spielraum, Optionen erforderlich",
    )
    faktenlage: list[Faktum] = Field(
        ...,
        description="Belegbare Aussagen mit Quellenbezug",
    )
    rechtsrahmen: list[Norm] = Field(
        default_factory=list,
        description="Einschlägige Rechtsnormen — leer lassen, wenn nicht juristisch",
    )
    optionen: list[Option] = Field(
        default_factory=list,
        description="2-4 Lösungsoptionen — NUR bei entscheidungsfrage befüllen",
    )
    vergleichsfaelle: list[str] = Field(
        default_factory=list,
        description="Empirische Vergleichsfälle (Nachbargemeinden, ähnliche Kommunen) mit kurzem Tier-Hinweis",
    )
    offene_fragen: list[str] = Field(
        default_factory=list,
        description="Was lässt sich aus den Quellen nicht beantworten?",
    )
    konfidenz: Literal["hoch", "mittel", "niedrig"] = Field(
        ...,
        description="Wie belastbar ist die Analyse insgesamt? Niedrig wenn nur Trainingswissen, hoch wenn Tier-1-belegt",
    )


def render_optionsanalyse(a: Optionsanalyse, sources: list[str]) -> str:
    """Optionsanalyse als lesbares Markdown rendern. sources: source-URL pro Chunk-Index."""
    lines = []

    lines.append(f"**Thema:** {a.thema}")
    lines.append(f"**Frage-Typ:** {a.frage_typ}  |  **Konfidenz:** {a.konfidenz}")
    lines.append("")

    lines.append("### 1. Faktenlage")
    if not a.faktenlage:
        lines.append("(keine belastbaren Fakten verfügbar)")
    status_marker = {
        "verifiziert": "✅",
        "teilweise": "🟡",
        "nicht_belegt": "⚪",
        "widersprochen": "❌",
    }
    for f in a.faktenlage:
        marker = status_marker.get(
            f.verification_status,
            "✓" if f.verifiziert else "○",  # Fallback wenn Stage 3 nicht lief
        )
        if f.quelle_chunk is not None and 0 <= f.quelle_chunk < len(sources):
            tag = f"Tier {f.tier} [chunk {f.quelle_chunk}]" if f.tier else f"chunk {f.quelle_chunk}"
        else:
            tag = "Trainingswissen"
        lines.append(f"- {marker} {f.aussage}  _({tag})_")
        if f.evidence_quote:
            lines.append(f"    > {f.evidence_quote}")
    lines.append("")

    if a.rechtsrahmen:
        lines.append("### 2. Rechtsrahmen")
        for n in a.rechtsrahmen:
            cite = f" [chunk {n.quelle_chunk}]" if n.quelle_chunk is not None else ""
            lines.append(f"- **{n.bezeichnung}**{cite} — {n.relevanz}")
        lines.append("")

    if a.optionen:
        lines.append("### 3. Optionen mit Trade-offs")
        for i, opt in enumerate(a.optionen, 1):
            lines.append(f"**Option {chr(64+i)}: {opt.titel}**")
            lines.append(f"  {opt.beschreibung}")
            lines.append(f"  - Trade-offs:")
            for t in opt.trade_offs:
                lines.append(f"    - {t}")
            lines.append(f"  - Wertannahmen:")
            for w in opt.wertannahmen:
                lines.append(f"    - {w}")
            lines.append("")

    if a.vergleichsfaelle:
        lines.append("### 4. Vergleichsfälle")
        for v in a.vergleichsfaelle:
            lines.append(f"- {v}")
        lines.append("")

    if a.offene_fragen:
        lines.append("### 5. Offene Fragen")
        for q in a.offene_fragen:
            lines.append(f"- {q}")
        lines.append("")

    return "\n".join(lines)
