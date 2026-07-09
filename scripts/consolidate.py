#!/usr/bin/env python3
"""Konsolidierungs-Schritt — schließt die Prüf-Schleife.

Bisher endete die Pipeline bei der Transparenz: sie zeigte „wir behaupten X" UND
„ein Modell hat X widerlegt" nebeneinander → Selbstwiderspruch, der die klare
Botschaft zersägt. Die Konsolidierung reagiert auf die Befunde (Adversarial +
Critique) und stellt eine in sich WIDERSPRUCHSFREIE Fassung her — voll automatisch.

Regeln (Robert, 2026-07-08):
  - FAKTENFEHLER (falsche/unbelegte Ist-Aussage, erfundene Norm/Zahl):
    korrigieren, wenn Quellen/Befunde die richtige Angabe hergeben; sonst die
    konkrete Angabe durch „(zu prüfen)" ersetzen. Nichts dazu erfinden.
  - FORDERUNG / WERTENTSCHEIDUNG (soll/wollen/Ziel — kein Faktenstreit):
    behalten, aber klar als Wert-/Forderungsentscheidung kennzeichnen.
  - UNHALTBARE IST-BEHAUPTUNG (weder belegbar noch als Forderung zu retten):
    NICHT löschen → zur OFFENEN FRAGE herabstufen („offen: …").

Erzeuger-Familie (Mistral) revidiert, damit Stil/Kernbotschaft erhalten bleiben.
"""
import json
import re

RULES = """Am Ende darf KEINE Aussage behauptet werden, die die Prüfung widerlegt hat. Löse jeden Befund so auf:
- FAKTENFEHLER (falsche/unbelegte Ist-Aussage, erfundene Norm-Nummer, falsche Zahl): korrigiere sie, wenn Quellen/Befunde die richtige Angabe hergeben; sonst ersetze die konkrete Zahl/Norm durch „(genaue Angabe zu prüfen)". Erfinde nichts.
- FORDERUNG / WERTENTSCHEIDUNG (soll/wollen/Ziel — kein Faktenstreit): behalten, aber klar als Wert-/Forderungsentscheidung kennzeichnen, nicht als Tatsache.
- UNHALTBARE IST-BEHAUPTUNG (weder belegbar noch als Forderung zu retten): NICHT löschen — von einer Behauptung zur OFFENEN FRAGE herabstufen (im Text als „offen: …").
Ändere nur, was die Befunde betreffen. Kernbotschaft und Ton bleiben erhalten."""

_NORM_RE = re.compile(r"§+\s?\d+[a-z]?(?:\s?Abs\.?\s?\d+[a-z]?)?(?:\s?(?:SGB|StGB|BGB|GG|EStG|EFZG)\b\.?(?:\s[IVX]+)?)?"
                      r"|Art\.?\s?\d+[a-z]?\s?(?:GG|AEUV|EUV|GRCh)\b")


def _parse(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        raise ValueError("kein JSON in der Konsolidierungs-Antwort")
    return json.loads(m.group(0))


def build_findings(adversarial=None, critique="", critique_verdict=None):
    """Bündelt Adversarial-Einwände + Critique(-Verdikt) zu einem Befund-Text."""
    parts = []
    for e in (adversarial or []):
        dim = e.get("dimension", "")
        parts.append(f"[Adversarial{('/' + dim) if dim else ''}] {e.get('einwand', '')}"
                     f"{(' → ' + e['empfohlene_ampel']) if e.get('empfohlene_ampel') else ''}")
    cv = critique_verdict or {}
    if cv:
        if cv.get("erfundene_fakten"):
            parts.append(f"[Critique] Nicht belegte/erfundene Fakten festgestellt: {cv.get('fazit', '')}")
        elif cv.get("fazit"):
            parts.append(f"[Critique] {cv.get('fazit', '')}")
        for h in (cv.get("hinweise") or [])[:6]:
            parts.append(f"[Critique-Hinweis] {h}")
    elif critique:
        parts.append(f"[Critique] {critique[:900]}")
    return "\n".join(p for p in parts if p.strip())


def consolidate_text(text, findings, llm, art="Position"):
    """Prosa-Konsolidierung (z. B. Programmpositionen). Liefert (konsolidiert, aufloesungen)."""
    if not (findings or "").strip():
        return text, []
    prompt = (f"Du konsolidierst eine {art}, sodass sie in sich WIDERSPRUCHSFREI ist.\n{RULES}\n\n"
              f"URSPRÜNGLICHER TEXT:\n{text[:6000]}\n\n"
              f"BEFUNDE (Adversarial + Critique):\n{findings[:4000]}\n\n"
              'Gib NUR JSON: {"konsolidiert":"<überarbeiteter Text>","aufloesungen":'
              '[{"befund":"<kurz>","art":"korrigiert|forderung|offene_frage|gestrichen","was":"<1 Satz>"}]}')
    obj = _parse(getattr(llm.invoke(prompt), "content", ""))
    return obj.get("konsolidiert", text) or text, obj.get("aufloesungen", [])


def strip_unsourced_norms(text, source_text):
    """Deterministisch: Norm-Nummern (§ …, Art. …), die NICHT in den Quellen stehen, durch
    „(Rechtsgrundlage zu prüfen)" ersetzen. Fängt erfundene Paragraphen ohne LLM-Risiko.
    Liefert (bereinigter_text, [entfernte_normen])."""
    if not text:
        return text, []
    src = re.sub(r"\s+", " ", source_text or "")
    removed = []

    def _norm_key(s):
        return re.sub(r"\s+", "", s).lower()

    src_norm = _norm_key(src)

    def repl(m):
        frag = m.group(0).strip(" .,;")
        # Kern der Norm (§ 92 Abs. 4a bzw. Art. 5) — reicht als Beleg-Check
        core = re.match(r"(§+\s?\d+[a-z]?(?:\s?Abs\.?\s?\d+[a-z]?)?|Art\.?\s?\d+[a-z]?)", frag)
        key = _norm_key(core.group(0)) if core else _norm_key(frag)
        if key and key in src_norm:
            return m.group(0)
        removed.append(frag)
        return "(Rechtsgrundlage zu prüfen)"

    out = _NORM_RE.sub(repl, text)
    return out, removed


def consolidate_analysis(analysis, source_text, llm, findings=""):
    """Konsolidiert eine strukturierte Umsetzungsanalyse (dict): Mistral prüft die
    Rechtsnormen KONTEXTUELL (Quellen + Critique-Funde + Fachwissen) und nennt nur die
    tatsächlich FALSCHEN/erfundenen; korrekte Normen (z. B. § 92 SGB V) bleiben. Die
    Korrekturen werden deterministisch als String-Ersatz angewandt, sodass Ampeln/
    Struktur unangetastet bleiben. Liefert (bereinigte_analyse, aufloesungen)."""
    fields = []
    for d in analysis.get("dimensionen", []):
        fields += [d.get("begruendung", ""), d.get("blocker", "")]
    fields += list(analysis.get("schritte", []))
    fields += list(analysis.get("betroffene", {}).get("gesetze", []))
    joined = "\n".join(t for t in fields if t)
    norms = sorted({m.strip() for m in _NORM_RE.findall(joined) if m.strip()})
    if not norms or llm is None:
        return analysis, []

    prompt = ("Prüfe die folgenden Rechtsnormen aus einer politischen Umsetzungsanalyse auf KONTEXTUELLE "
              "Richtigkeit — gestützt auf die Quellen und dein juristisches Fachwissen. Nenne NUR Normen, die "
              "für diese Maßnahme FALSCH zugeordnet oder erfunden sind. Korrekte Normen NICHT anfassen.\n"
              "Für jede falsche Norm: die korrekte Norm angeben, wenn du sie sicher weißt; sonst wörtlich "
              "\"Rechtsgrundlage zu prüfen\".\n\n"
              "NORMEN:\n" + "\n".join(f"- {n}" for n in norms) + "\n\n"
              f"QUELLEN:\n{source_text[:4000]}\n\n"
              f"CRITIQUE-FUNDE:\n{findings[:1500]}\n\n"
              'Gib NUR JSON: {"korrekturen":[{"alt":"<Norm exakt wie oben>","neu":"<korrekte Norm | Rechtsgrundlage zu prüfen>"}]}')
    try:
        obj = _parse(getattr(llm.invoke(prompt), "content", ""))
    except Exception:
        return analysis, []
    korr = [(k.get("alt", "").strip(), (k.get("neu", "").strip() or "Rechtsgrundlage zu prüfen"))
            for k in obj.get("korrekturen", []) if k.get("alt", "").strip()]
    if not korr:
        return analysis, []

    def fix(s):
        for alt, neu in korr:
            if alt and alt in s:
                s = s.replace(alt, neu)
        return s

    a = json.loads(json.dumps(analysis, ensure_ascii=False))  # tiefe Kopie
    for d in a.get("dimensionen", []):
        d["begruendung"] = fix(d.get("begruendung", ""))
        d["blocker"] = fix(d.get("blocker", ""))
    a["schritte"] = [fix(s) for s in a.get("schritte", [])]
    b = a.get("betroffene", {})
    b["gesetze"] = [fix(s) for s in b.get("gesetze", [])]
    aufl = [{"befund": f"Norm {alt} falsch/unbelegt", "art": "korrigiert", "was": f"→ {neu}"} for alt, neu in korr]
    return a, aufl
