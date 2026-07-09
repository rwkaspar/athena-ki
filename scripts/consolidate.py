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


def strip_fabrications(analysis, source_text, llm, findings=""):
    """Aktuator gegen Fabrikation: findet die im Text stehenden EMPIRISCHEN Behauptungen
    (Zahlen, €-Beträge, %, Statistiken, Normen, dritten Parteien zugeschriebene Positionen,
    Fremdthemen), die WEDER in den Quellen NOCH im Vorhaben belegt sind, und MARKIERT sie
    inline als ⟦unbelegt: …⟧ — nicht löschen (könnte korrekt-aber-unretrievt sein), sondern
    für die Handprüfung sichtbar machen. Liefert (markierte_analyse, fabrikate[])."""
    if llm is None:
        return analysis, []
    # Prosa-Felder einsammeln, die empirische Behauptungen tragen können
    fields = []
    z = analysis.get("zielerreichung", {})
    fields.append(z.get("begruendung", ""))
    for nw in analysis.get("nebenwirkungen", []):
        fields += [nw.get("effekt", ""), nw.get("begruendung", "")]
    fields.append((analysis.get("kohaerenz") or {}).get("begruendung", ""))
    fields.append((analysis.get("verteilung") or {}).get("begruendung", ""))
    for d in analysis.get("dimensionen", []):
        fields += [d.get("begruendung", ""), d.get("blocker", "")]
    fields.append((analysis.get("verhaeltnismaessigkeit") or {}).get("begruendung", ""))
    joined = "\n".join(t for t in fields if t)
    if not joined.strip():
        return analysis, []

    prompt = ("Prüfe die folgende politische Analyse auf ERFUNDENE empirische Behauptungen. Liste JEDE "
              "Aussage, die eine konkrete Zahl, einen €-Betrag, einen Prozentwert, eine Statistik, eine "
              "Rechtsnorm/Paragraphen, eine einer dritten Partei zugeschriebene Position ODER ein Thema "
              "nennt, das WEDER in den QUELLEN NOCH im VORHABEN belegt ist. Gib den betreffenden Textteil "
              "WÖRTLICH und EXAKT so zurück, wie er in der Analyse steht (kopierfähige Teil-Zeichenkette).\n"
              "NICHT listen: qualitatives Schließen, Wertungen, Forderungen, Ampel-Begriffe. NUR unbelegte "
              "empirische FAKTEN.\n\n"
              f"VORHABEN + QUELLEN:\n{source_text[:5000]}\n\n"
              f"HINWEISE DES REVIEWERS:\n{findings[:1500]}\n\n"
              f"ANALYSE-TEXT:\n{joined[:5000]}\n\n"
              'Gib NUR JSON: {"fabrikate":[{"zitat":"<wörtliche Teil-Zeichenkette aus der Analyse>","grund":"<kurz, warum unbelegt>"}]}')
    try:
        obj = _parse(getattr(llm.invoke(prompt), "content", ""))
    except Exception:
        return analysis, []
    fab = [(f.get("zitat", "").strip(), f.get("grund", "").strip())
           for f in obj.get("fabrikate", []) if len(f.get("zitat", "").strip()) >= 4]
    if not fab:
        return analysis, []

    def mark(s):
        for zitat, _ in fab:
            if zitat and zitat in s and "⟦unbelegt:" not in s[max(0, s.find(zitat) - 12):s.find(zitat)]:
                s = s.replace(zitat, f"⟦unbelegt: {zitat}⟧", 1)
        return s

    a = json.loads(json.dumps(analysis, ensure_ascii=False))  # tiefe Kopie
    za = a.get("zielerreichung") or {}
    za["begruendung"] = mark(za.get("begruendung", ""))
    for nw in a.get("nebenwirkungen", []):
        nw["effekt"] = mark(nw.get("effekt", ""))
        nw["begruendung"] = mark(nw.get("begruendung", ""))
    for key in ("kohaerenz", "verteilung", "verhaeltnismaessigkeit"):
        d = a.get(key)
        if isinstance(d, dict):
            d["begruendung"] = mark(d.get("begruendung", ""))
    for d in a.get("dimensionen", []):
        d["begruendung"] = mark(d.get("begruendung", ""))
        d["blocker"] = mark(d.get("blocker", ""))
    fabrikate = [{"zitat": z, "grund": g} for z, g in fab]
    return a, fabrikate


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
    # Deterministischer Beleg-Check: Normen, die WÖRTLICH im Vorhaben/in den Quellen stehen,
    # sind per Definition belegt (das Paper nennt sie selbst) → gar nicht erst zur Prüfung geben.
    # Sonst „korrigiert" das Modell mit seinem Fachwissen eine korrekte, quellengenannte Norm
    # (z. B. § 278 StGB) in eine falsche. Nur wirklich UNbelegte Normen gehen an das LLM.
    src_key = re.sub(r"\s+", "", source_text or "").lower()
    norms = [n for n in norms if re.sub(r"\s+", "", n).lower() not in src_key]
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
    def _clean_neu(neu):
        """Die Ersetzung muss eine KURZE, saubere Normangabe bleiben — kein Aufsatz.
        Verbose Begründungen (Klammern, Semikolons, > 45 Zeichen) auf die Standardfloskel
        zurückstutzen, damit die gesetze-/begruendung-Felder nicht mit Prosa vermüllt werden."""
        neu = (neu or "").strip()
        if not neu or len(neu) > 45 or "(" in neu or ";" in neu:
            return "Rechtsgrundlage zu prüfen"
        return neu

    korr = [(k.get("alt", "").strip(), _clean_neu(k.get("neu", "")))
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
