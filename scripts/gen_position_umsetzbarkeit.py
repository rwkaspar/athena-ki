#!/usr/bin/env python3
"""Wirkungs- & Umsetzungsanalyse — prüft ein VORHABEN in ZWEI getrennten Analysen:

  WIRKUNGSANALYSE  — erreicht die Maßnahme ihr erklärtes ZIEL? (empirisch, zielerreichung ja/teilweise/nein)
  UMSETZUNGSANALYSE — lässt sie sich DURCHFÜHREN? (4 Dimensionen rechtlich/administrativ/
     politisch/finanziell als Ampel + Blocker, Rahmen/Schritte, Dauer, Betroffene)

Beide getrennt halten — eine Maßnahme kann voll umsetzbar sein und ihr Ziel trotzdem verfehlen.
3-Familien-Prüfung wie die Hauptpipeline: Erzeugung=Mistral (mit RAG faktentreu), Adversarial=gemma4
(greift die Ampeln an), Critique=athena (holistischer Review gegen die Quellen, fängt Fabrikationen).

ZWEI Modi:
  Positions-Modus (default): analysiert die 14 EVIDENZ-Programmpositionen → schreibt
    evidenz-partei/data/umsetzbarkeit.json (für die Positionsseiten).
  Standalone-Modus (--reform "…" | --reform-file DATEI): analysiert ein beliebiges
    Vorhaben/eine Reform → Ausgabe menschenlesbar + optional JSON (--out).

Aufruf:
  OLLAMA_HOST=… python scripts/gen_position_umsetzbarkeit.py [--only slug] [--limit N]
  OLLAMA_HOST=… python scripts/gen_position_umsetzbarkeit.py --reform "Abschaffung der …" --out x.json
"""
import argparse, datetime, json, os, re, sys, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_MD = ROOT / "output"
UM_JSON = ROOT.parent / "evidenz-partei" / "data" / "umsetzbarkeit.json"

POSITIONS = [
    ("parteiprogramm_01_direktdemokratie.md", "direkte-demokratie"),
    ("parteiprogramm_02_wahlrecht.md", "wahlrecht"),
    ("parteiprogramm_03_parteienfinanzierung.md", "parteienfinanzierung"),
    ("parteiprogramm_04_schuldenbremse.md", "schuldenbremse"),
    ("parteiprogramm_05_steuersystem.md", "steuersystem"),
    ("parteiprogramm_05a_ki_regulation.md", "ki-regulation"),
    ("parteiprogramm_06_buergergeld.md", "buergergeld"),
    ("parteiprogramm_07_parlamentarismus.md", "parlamentarismus"),
    ("parteiprogramm_07a_abgeordnete_diaeten.md", "abgeordnete-diaeten"),
    ("parteiprogramm_08_rente.md", "rente"),
    ("parteiprogramm_09_klima_energie.md", "klima-energie"),
    ("parteiprogramm_10_migration_asyl.md", "migration-asyl"),
    ("parteiprogramm_11_verteidigung.md", "verteidigung"),
    ("parteiprogramm_12_gesundheit_buergerversicherung.md", "gesundheit"),
]

DIMENSIONEN = ["rechtlich", "administrativ", "politisch", "finanziell"]
AMPEL = ["hoch", "mittel", "gering", "nicht_umsetzbar"]      # von gut → schlecht
RANK = {a: i for i, a in enumerate(AMPEL)}                    # größer = schlechter
KOSTEN_STUFEN = ["symbolisch", "niedrig", "mittel", "hoch"]

GEN_PROMPT = """Heute ist der {heute}. Bewerte auf diesem Stand; für ein erst kürzlich beschlossenes \
Vorhaben liegt in aller Regel noch KEINE Wirkungsevidenz vor — das ist dann „unbelegt", keine Schwäche.
Du analysierst, wie realistisch ein konkretes politisches VORHABEN \
tatsächlich UMSETZBAR ist — nüchtern, ehrlich, nicht werbend. Bewerte die Umsetzung im \
deutschen Regierungs-/Verwaltungssystem, gestützt auf die beigegebenen Quellen. Das gilt \
für Vorhaben JEDER Herkunft (Parteiposition, Regierungsreform, Gesetzentwurf) — du bewertest \
nur die Machbarkeit, nicht ob das Ziel wünschenswert ist.

WICHTIG — halte DREI Ebenen strikt getrennt (nicht vermengen):
- A) WIRKUNG: Taugt die Maßnahme inhaltlich? (Ziel erreicht, Nebenwirkungen, Kohärenz, Verteilung)
- B) UMSETZUNG: Lässt sie sich im System DURCHFÜHREN? (nur Machbarkeit)
- C) ABWÄGUNG: Verhältnismäßigkeit + Alternativen.
Eine Maßnahme kann voll umsetzbar sein und inhaltlich trotzdem durchfallen — ein Wirkungsproblem darf NIE
als „nicht umsetzbar" verbucht werden. Und umgekehrt.
KEINE Empfehlung: Du BEWERTEST, sprichst aber KEINE Handlungsempfehlung aus (weder für noch gegen). Wo eine
Entscheidung an einer Wertfrage hängt, benenne die Wertannahmen statt zu empfehlen.

════ QUELLENDISZIPLIN (wichtigste Regel — Verstoß macht die Analyse wertlos) ════
Jede EMPIRISCHE Tatsachenbehauptung — Zahl, Betrag, Statistik, Prozentwert, Rechtsnorm/Paragraph, sowie jede
einer dritten Partei zugeschriebene Position (BDA, Gewerkschaft, Behörde …) — MUSS entweder in den QUELLEN
oder im VORHABEN-Text stehen. Steht sie dort nicht, ERFINDE sie NICHT: lass sie weg oder schreibe wörtlich
„keine belastbare Quelle". Erfinde KEINE Kostenschätzungen („ca. 1–2 Mrd. €"), KEINE Paragraphen, KEINE
Statistiken. Bring auch KEINE Fremdthemen ein, die nicht in dieser Maßnahme stehen.
ERLAUBT ist qualitatives Schließen AUS der Maßnahme selbst (logische Folgen, Zielkonflikte, betroffene Gruppen).
VERBOTEN sind erfundene empirische Fakten. Lieber eine Achse ehrlich „unbelegt" als mit Erfindung gefüllt.

════ A) WIRKUNGSANALYSE ════
(A0) zielerreichung: Erreicht die Maßnahme ihr ERKLÄRTES Ziel? Wähle den Status STRENG nach der Evidenzlage:
     - "nein"      = die Evidenz WIDERLEGT die Prämisse/Wirkmechanismus (das Problem existiert so nicht, oder der Mechanismus wirkt nachweislich nicht).
     - "ja"        = die Quellen STÜTZEN, dass der Mechanismus das Ziel erreicht.
     - "teilweise" = die Evidenz ist gemischt (stützt teils, widerlegt teils) — nur bei ECHTEM Mischbefund.
     - "unbelegt"  = zu dieser konkreten Wirkung liegt KEINE belastbare Evidenz vor (ehrlicher Default für neue Maßnahmen ohne Wirkungsnachweis — NICHT „teilweise" ausweichen).
     + begruendung (1 Satz; bei "ja"/"nein"/"teilweise" mit Beleg, bei "unbelegt" die Evidenzlücke benennen).
(A1) nebenwirkungen: 0–4 unbeabsichtigte Folgen JENSEITS des Ziels — NUR belegte oder logisch ZWINGENDE.
     Erfinde keine, um die Liste zu füllen; leere Liste ist erlaubt. je {effekt, richtung "schaedlich"|"neutral"|"positiv", begruendung}.
(A2) kohaerenz: Widerspricht die Maßnahme anderen Zielen — insbesondere den EIGENEN Zielen des Pakets/Programms?
     status "kohaerent"|"spannung"|"widerspruch" + begruendung.
(A3) verteilung: Wer trägt den Nutzen, wer die Last? gewinner[], verlierer[], begruendung (1 Satz).

════ B) UMSETZUNGSANALYSE ════
(B0) dimensionen: VIER Dimensionen, je als Ampel — NUR Machbarkeit, nicht Wirkung:
     "hoch"=ohne große Hürden machbar · "mittel"=machbar mit erheblichem Aufwand · "gering"=nur mit großen Hürden
     (⅔-Mehrheit, EU-Einigung, langer Verwaltungsaufbau) · "nicht_umsetzbar"=praktisch nicht durchführbar (dann Blocker).
     - rechtlich = welche Rechtsänderung nötig, verfassungs-/EU-rechtlich haltbar?
     - administrativ = schaffen Verwaltung UND KAPAZITÄTEN (Praxen, Behörden, Personal) das real? Kapazitätsengpässe = Blocker.
     - politisch = ist die Mehrheit da, wie groß der Widerstand?
     - finanziell = FINANZIERBAR/bezahlbar? NICHT ob das Ziel erreicht wird. Was Geld kostet, ist finanziell umsetzbar —
       nur echte Unfinanzierbarkeit (z. B. „Rasenmäher" scheitert an gebundenen Ausgaben) ist „nicht_umsetzbar".
(B1) schritte: 3–6 konkrete Umsetzungsschritte in Reihenfolge (Gesetz, Verordnung, GG-Änderung, EU-Ebene, Verwaltungsaufbau …).
(B2) umsetzungsdauer "kurzfristig"|"mittelfristig"|"langfristig"; dauer_jahre (Text, z. B. „1–2 Jahre").
(B3) betroffene: gesetze/Normen, institutionen, gruppen, kosten ("symbolisch"|"niedrig"|"mittel"|"hoch"|"unbekannt" + Halbsatz).
     Rechtsgrundlagen/Paragraphen NUR nennen, wenn sie in den QUELLEN ODER im VORHABEN stehen — sonst „Rechtsgrundlage zu prüfen" statt Norm erfinden.
     kosten NUR als Stufe schätzen, wenn die Quellen/das Vorhaben das hergeben; sonst "unbekannt". KEINE erfundenen Euro-Beträge.
Außerdem: reversibilitaet ("leicht"|"mittel"|"schwer"|"irreversibel"), konfidenz ("hoch"|"mittel"|"niedrig").

════ C) ABWÄGUNG ════
(C0) verhaeltnismaessigkeit: Steht der Aufwand/die Kosten im Verhältnis zum tatsächlichen Nutzen?
     status "angemessen"|"fraglich"|"unverhaeltnismaessig" + begruendung.
(C1) alternativen: 1–4 Optionen mit je {option, begruendung}. WICHTIG: Ist die PRÄMISE/Grundlage der Maßnahme
     widerlegt, ist die beste Option oft „Maßnahme ersatzlos streichen" — erfinde KEINE Ersatzmaßnahme, nur um eine zu haben.
     Sprich KEINE Empfehlung aus, welche Alternative zu wählen ist.

Antworte NUR als JSON, exakt in dieser Form:
{"zielerreichung":{"status":"ja|teilweise|nein|unbelegt","begruendung":"<1 Satz>"},
 "nebenwirkungen":[{"effekt":"<…>","richtung":"schaedlich|neutral|positiv","begruendung":"<…>"}],
 "kohaerenz":{"status":"kohaerent|spannung|widerspruch","begruendung":"<…>"},
 "verteilung":{"gewinner":["<…>"],"verlierer":["<…>"],"begruendung":"<…>"},
 "dimensionen":[{"dimension":"rechtlich","ampel":"<ampel>","begruendung":"<1 Satz>","blocker":"<nur bei gering/nicht_umsetzbar>"},
 {"dimension":"administrativ",…},{"dimension":"politisch",…},{"dimension":"finanziell",…}],
 "schritte":["…"],"abhaengigkeiten":["…"],
 "umsetzungsdauer":"<…>","dauer_jahre":"<…>","reversibilitaet":"<…>","konfidenz":"<…>",
 "betroffene":{"gesetze":["…"],"institutionen":["…"],"gruppen":["…"],"kosten":"<stufe> — <halbsatz>"},
 "verhaeltnismaessigkeit":{"status":"angemessen|fraglich|unverhaeltnismaessig","begruendung":"<…>"},
 "alternativen":[{"option":"<… (auch: Maßnahme ersatzlos streichen)>","begruendung":"<…>"}]}

QUELLEN (RAG, mit Tier-Hinweis):
{context}

VORHABEN (zu prüfen):
{vorhaben}
"""

ADV_PROMPT = """Hier ist ein politisches Vorhaben und eine Umsetzbarkeits-Einschätzung dazu. \
Deine Aufgabe: die Einschätzung ADVERSARIAL angreifen — sei der Advocatus Diaboli der Machbarkeit. \
Prüfe jede der vier Dimensionen (rechtlich, administrativ, politisch, finanziell): Wo ist die Ampel \
zu GRÜN/optimistisch? Welche konkrete Hürde wurde übersehen? Denke an Fälle wie die pauschale \
„Rasenmähermethode" oder Maßnahmen, deren Ziel die Evidenz nicht stützt.

Melde NUR Dimensionen, bei denen du eine STRENGERE Ampel für richtig hältst. Für jede: der konkrete \
Einwand (1–2 Sätze) und die aus deiner Sicht korrekte Ampel (hoch/mittel/gering/nicht_umsetzbar).
Wenn die Einschätzung realistisch ist, gib eine leere Liste zurück.

Antworte NUR als JSON:
{"einwaende":[{"dimension":"<rechtlich|administrativ|politisch|finanziell>","einwand":"<1-2 Sätze>","empfohlene_ampel":"<ampel>"}]}

UMSETZBARKEITS-EINSCHÄTZUNG (zu prüfen):
{einschaetzung}

VORHABEN:
{vorhaben}
"""


def _parse_json(raw):
    return json.loads(re.search(r"\{.*\}", raw, re.S).group(0))


def _norm_ampel(a, default="mittel"):
    a = (a or "").strip().lower().replace(" ", "_").replace("-", "_")
    return a if a in RANK else default


def _clean_list(x, n=6, cap=200):
    return [str(s).strip()[:cap] for s in (x or []) if str(s).strip()][:n]


def analyze(vorhaben: str, query: str, vs, gen, adv, critique, host, serve, weltkontext: str = "", heute: str = ""):
    """Kern (3-Familien wie die Hauptpipeline): Retrieval → Erzeugung MISTRAL →
    gemma4-Adversarial (Ampeln) → Merge → athena-CRITIQUE (holistischer Review +
    Verdikt, fängt Fabrikationen wie falsche Norm-Nummern). Liefert (rec, docs)."""
    from retrieval import tier_aware_retrieve, format_docs
    from pipeline_demo import critique_verdict
    docs = tier_aware_retrieve(vs, query, k=serve.RETRIEVER_K, fetch_k=50, sim_floor=0.4, max_k=12)
    ctx = format_docs(docs)
    # Erzeugung: Mistral (mit RAG faktentreu) — ChatMistralAI liefert eine Message.
    resp = gen.invoke(GEN_PROMPT.replace("{context}", ctx).replace("{vorhaben}", vorhaben[:8000])
                      .replace("{heute}", heute or "heute"))
    raw = getattr(resp, "content", resp)
    u = _parse_json(raw)

    dims = {}
    for d in u.get("dimensionen", []):
        name = (d.get("dimension") or "").strip().lower()
        if name in DIMENSIONEN:
            dims[name] = {"dimension": name, "ampel": _norm_ampel(d.get("ampel")),
                          "begruendung": (d.get("begruendung") or "").strip()[:220],
                          "blocker": (d.get("blocker") or "").strip()[:220]}
    for name in DIMENSIONEN:
        dims.setdefault(name, {"dimension": name, "ampel": "mittel",
                               "begruendung": "Nicht eingeschätzt.", "blocker": ""})

    einwaende = []
    try:
        einschaetzung = json.dumps({"dimensionen": list(dims.values())}, ensure_ascii=False)
        raw2 = adv.invoke(ADV_PROMPT.replace("{einschaetzung}", einschaetzung).replace("{vorhaben}", vorhaben[:6000]))
        for e in _parse_json(raw2).get("einwaende", []):
            name = (e.get("dimension") or "").strip().lower()
            if name in DIMENSIONEN:
                einwaende.append({"dimension": name, "einwand": (e.get("einwand") or "").strip()[:280],
                                  "empfohlene_ampel": _norm_ampel(e.get("empfohlene_ampel"))})
    except Exception as e:
        print(f"  [warn] adversarial fehlgeschlagen: {type(e).__name__}", file=sys.stderr)

    # Merge: auf die konservativere (schlechtere) Ampel herunterstufen, Einwand sichtbar lassen
    for e in einwaende:
        d = dims[e["dimension"]]
        if RANK[e["empfohlene_ampel"]] > RANK[d["ampel"]]:
            d["ampel"] = e["empfohlene_ampel"]
            zusatz = f"Adversarial: {e['einwand']}"
            d["blocker"] = (d["blocker"] + " · " + zusatz) if d["blocker"] else zusatz

    dim_list = [dims[n] for n in DIMENSIONEN]
    gesamt_rank = max((RANK[d["ampel"]] for d in dim_list), default=RANK["mittel"])
    b = u.get("betroffene") or {}
    kosten = (b.get("kosten") or "").strip()[:160]
    # Wirksamkeit (Zielerreichung) — eigene Achse, getrennt von der Machbarkeit.
    z = u.get("zielerreichung") or {}
    ziel = (z.get("status") or "").strip().lower()
    if ziel not in ("ja", "teilweise", "nein", "unbelegt"):
        ziel = "unbelegt"   # ehrlicher Default: keine Wirkungsevidenz ≠ „teilweise"
    UMSETZ_LABEL = {"hoch": "gut umsetzbar", "mittel": "umsetzbar mit Aufwand",
                    "gering": "umsetzbar, aber mit großen Hürden", "nicht_umsetzbar": "nicht umsetzbar"}
    umsetz_label = UMSETZ_LABEL[AMPEL[gesamt_rank]]
    fazit = (f"{umsetz_label} — verfehlt aber das erklärte Ziel." if ziel == "nein"
             else f"{umsetz_label}; erreicht das Ziel nur teilweise." if ziel == "teilweise"
             else f"{umsetz_label}; Zielerreichung ist mangels Evidenz offen." if ziel == "unbelegt"
             else f"{umsetz_label} und zielführend.")
    # Weitere Wirkungs-Achsen
    nebenwirkungen = []
    for nw in (u.get("nebenwirkungen") or [])[:4]:
        eff = (nw.get("effekt") or "").strip()[:200]
        if eff:
            r = (nw.get("richtung") or "neutral").strip().lower()
            nebenwirkungen.append({"effekt": eff, "richtung": r if r in ("schaedlich", "neutral", "positiv") else "neutral",
                                   "begruendung": (nw.get("begruendung") or "").strip()[:220]})
    koh = u.get("kohaerenz") or {}
    koh_s = (koh.get("status") or "").strip().lower()
    kohaerenz = {"status": koh_s if koh_s in ("kohaerent", "spannung", "widerspruch") else "spannung",
                 "begruendung": (koh.get("begruendung") or "").strip()[:280]}
    vt = u.get("verteilung") or {}
    verteilung = {"gewinner": _clean_list(vt.get("gewinner"), n=6, cap=100),
                  "verlierer": _clean_list(vt.get("verlierer"), n=6, cap=100),
                  "begruendung": (vt.get("begruendung") or "").strip()[:220]}
    vh = u.get("verhaeltnismaessigkeit") or {}
    vh_s = (vh.get("status") or "").strip().lower()
    verhaeltnismaessigkeit = {"status": vh_s if vh_s in ("angemessen", "fraglich", "unverhaeltnismaessig") else "fraglich",
                              "begruendung": (vh.get("begruendung") or "").strip()[:280]}
    alternativen = []
    for al in (u.get("alternativen") or [])[:4]:
        opt = (al.get("option") or "").strip()[:200]
        if opt:
            alternativen.append({"option": opt, "begruendung": (al.get("begruendung") or "").strip()[:220]})
    rec = {
        "zielerreichung": {"status": ziel, "begruendung": (z.get("begruendung") or "").strip()[:280]},
        "nebenwirkungen": nebenwirkungen,
        "kohaerenz": kohaerenz,
        "verteilung": verteilung,
        "gesamt": AMPEL[gesamt_rank],   # nur Machbarkeit (schlechteste Dimension)
        "verhaeltnismaessigkeit": verhaeltnismaessigkeit,
        "alternativen": alternativen,
        "fazit": fazit,
        "umsetzungsdauer": (u.get("umsetzungsdauer") or "mittelfristig").strip().lower(),
        "dauer_jahre": (u.get("dauer_jahre") or "").strip()[:40],
        "reversibilitaet": (u.get("reversibilitaet") or "mittel").strip().lower(),
        "konfidenz": (u.get("konfidenz") or "mittel").strip().lower(),
        "dimensionen": dim_list,
        "schritte": _clean_list(u.get("schritte")),
        "abhaengigkeiten": _clean_list(u.get("abhaengigkeiten")),
        "betroffene": {
            "gesetze": _clean_list(b.get("gesetze"), n=8, cap=120),
            "institutionen": _clean_list(b.get("institutionen"), n=8, cap=120),
            "gruppen": _clean_list(b.get("gruppen"), n=8, cap=120),
            "kosten": kosten,
        },
        "adversarial": einwaende,
        # Revisions-Hinweis: sobald eine Dimension oder das Gesamt „nicht umsetzbar" ist,
        # ist das Vorhaben in der Form überarbeitungsbedürftig (Mensch entscheidet).
        "revision_needed": AMPEL[gesamt_rank] == "nicht_umsetzbar"
                           or any(d["ampel"] == "nicht_umsetzbar" for d in dim_list)
                           or ziel == "nein",
        "quellen": [d.metadata.get("source") for d in docs if d.metadata.get("source")][:8],
    }

    # Critique (athena): holistischer methodischer Review der Analyse GEGEN die Quellen
    # + faires Verdikt. Fängt genau Fabrikationen wie falsche Norm-Nummern (§ 630d),
    # übersehene Dimensionen oder eine Verwechslung von Ziel und Maßnahme.
    try:
        crit = critique(query, docs, _analysis_to_text(rec, vorhaben), weltkontext=weltkontext, heute=heute)
        rec["critique"] = crit
        rec["critique_verdict"] = critique_verdict(crit, host)
    except Exception as e:
        print(f"  [warn] critique fehlgeschlagen: {type(e).__name__}", file=sys.stderr)
        rec["critique"] = ""
        rec["critique_verdict"] = {}
    # Fabrikationsverdacht: der Critique hat unbelegte/erfundene Detail-Behauptungen gefunden
    # (z. B. erfundene Kostenzahlen, falsch zugeschriebene Positionen). Die Norm-Konsolidierung
    # bereinigt nur Paragraphen — solche Behauptungen bleiben im Text und MÜSSEN vor Freigabe
    # von Hand geprüft werden. Als sichtbares Review-Flag führen.
    rec["fabrikationsverdacht"] = bool((rec.get("critique_verdict") or {}).get("erfundene_fakten"))

    # Konsolidierung — schließt die Schleife: erfundene Norm-Nummern (die der Critique
    # bemängelt) deterministisch gegen die Quellen bereinigen → „(Rechtsgrundlage zu prüfen)".
    try:
        from consolidate import consolidate_analysis, build_findings
        findings = build_findings(rec.get("adversarial"), rec.get("critique", ""), rec.get("critique_verdict"))
        # Das VORHABEN selbst zählt als Quelle: eine vom Paper GENANNTE Norm (z. B. § 278 StGB)
        # ist belegt und darf nicht als „unbelegt" umgeschrieben werden. Sonst macht die
        # Konsolidierung aus einer korrekten Norm eine falsche (§ 278 → § 277/279).
        cleaned, aufl = consolidate_analysis(rec, vorhaben + "\n\n" + ctx, gen, findings=findings)
        rec["dimensionen"] = cleaned["dimensionen"]
        rec["schritte"] = cleaned["schritte"]
        rec["betroffene"] = cleaned["betroffene"]
        rec["konsolidierung"] = aufl
        if aufl:
            print(f"  [konsolidiert] {len(aufl)} Norm(en) ohne Quellenbeleg auf 'zu pruefen' gesetzt", file=sys.stderr)
    except Exception as e:
        print(f"  [warn] konsolidierung fehlgeschlagen: {type(e).__name__}", file=sys.stderr)
        rec["konsolidierung"] = []

    # Aktuator: wenn der Critique unbelegte Fakten fand, die konkreten Behauptungen inline
    # als ⟦unbelegt: …⟧ markieren (nicht löschen — Handprüfung entscheidet). Schließt die
    # Lücke „Detektor ohne Aktuator": das Flag zählt nicht mehr nur, es zeigt WAS zu prüfen ist.
    rec["fabrikate"] = []
    if rec.get("fabrikationsverdacht"):
        try:
            from consolidate import strip_fabrications
            findings2 = build_findings(rec.get("adversarial"), rec.get("critique", ""), rec.get("critique_verdict"))
            # Weltkontext mitgeben, damit der Aktuator reale, belegte Angaben (Koalition, Datum)
            # nicht als „unbelegt" markiert.
            akt_src = (weltkontext + "\n\n" if weltkontext else "") + vorhaben + "\n\n" + ctx
            marked, fabrikate = strip_fabrications(rec, akt_src, gen, findings=findings2)
            for k in ("zielerreichung", "nebenwirkungen", "kohaerenz", "verteilung",
                      "verhaeltnismaessigkeit", "dimensionen"):
                if k in marked:
                    rec[k] = marked[k]
            rec["fabrikate"] = fabrikate
            if fabrikate:
                print(f"  [aktuator] {len(fabrikate)} unbelegte Behauptung(en) markiert", file=sys.stderr)
        except Exception as e:
            print(f"  [warn] aktuator fehlgeschlagen: {type(e).__name__}", file=sys.stderr)
    return rec, docs


def _analysis_to_text(rec, vorhaben):
    """Rendert die strukturierte Umsetzungsanalyse als prüfbaren Text für den Critique."""
    z = rec.get("zielerreichung", {})
    koh = rec.get("kohaerenz", {})
    vt = rec.get("verteilung", {})
    vh = rec.get("verhaeltnismaessigkeit", {})
    L = [f"VORHABEN: {vorhaben[:600]}",
         f"Wirkung — Ziel erreicht?: {z.get('status','')} — {z.get('begruendung','')}",
         "Nebenwirkungen: " + " | ".join(f"{nw['richtung']}: {nw['effekt']}" for nw in rec.get("nebenwirkungen", [])),
         f"Kohärenz/Zielkonflikt: {koh.get('status','')} — {koh.get('begruendung','')}",
         f"Verteilung: Gewinner [{', '.join(vt.get('gewinner',[]))}] Verlierer [{', '.join(vt.get('verlierer',[]))}]",
         f"Umsetzbarkeit gesamt: {rec['gesamt']} · Fazit: {rec.get('fazit','')}",
         f"Verhältnismäßigkeit: {vh.get('status','')} — {vh.get('begruendung','')}",
         "Alternativen: " + " | ".join(al['option'] for al in rec.get("alternativen", [])),
         f"Dauer: {rec['umsetzungsdauer']} ({rec['dauer_jahre']})"]
    for d in rec["dimensionen"]:
        L.append(f"- {d['dimension']}: {d['ampel']} — {d['begruendung']}"
                 + (f" [Blocker: {d['blocker']}]" if d["blocker"] else ""))
    if rec["schritte"]:
        L.append("Schritte: " + " | ".join(rec["schritte"]))
    b = rec["betroffene"]
    L.append("Betroffen — Gesetze: " + ", ".join(b["gesetze"]) + "; Institutionen: "
             + ", ".join(b["institutionen"]) + "; Gruppen: " + ", ".join(b["gruppen"])
             + "; Kosten: " + b["kosten"])
    return "\n".join(L)


def _print_human(titel, rec):
    ICON = {"hoch": "🟢", "mittel": "🟡", "gering": "🟠", "nicht_umsetzbar": "🔴"}
    ZICON = {"ja": "🟢", "teilweise": "🟡", "nein": "🔴", "unbelegt": "⚪"}
    z = rec.get("zielerreichung", {})
    print(f"\n╔══ ANALYSE: {titel[:70]}")
    # Zwei getrennte Analysen — nicht unter einen Scheffel.
    RICON = {"schaedlich": "🔴", "neutral": "⚪", "positiv": "🟢"}
    print(f"║ ▓▓ WIRKUNGSANALYSE")
    print(f"║   Ziel erreicht? {ZICON.get(z.get('status'),'')} {(z.get('status') or '').upper()} — {z.get('begruendung','')}")
    for nw in rec.get("nebenwirkungen", []):
        print(f"║   Nebenwirkung {RICON.get(nw['richtung'],'')} {nw['effekt']} — {nw['begruendung']}")
    koh = rec.get("kohaerenz", {})
    print(f"║   Kohärenz/Zielkonflikt: {koh.get('status','')} — {koh.get('begruendung','')}")
    vt = rec.get("verteilung", {})
    print(f"║   Verteilung: Gewinner [{', '.join(vt.get('gewinner',[]))}] · Verlierer [{', '.join(vt.get('verlierer',[]))}]")
    print(f"║ ▓▓ UMSETZUNGSANALYSE  {ICON.get(rec['gesamt'],'')} {rec['gesamt']}"
          f"  · Dauer: {rec['umsetzungsdauer']} ({rec['dauer_jahre']}) · Reversibilität: {rec['reversibilitaet']}")
    print("║ ── je Dimension ──")
    for d in rec["dimensionen"]:
        print(f"║   {ICON.get(d['ampel'],'')} {d['dimension']:12} {d['ampel']:16} {d['begruendung']}")
        if d["blocker"]:
            print(f"║       ⛔ {d['blocker']}")
    print("║ ── Rahmen / Schritte ──")
    for s in rec["schritte"]:
        print(f"║   → {s}")
    b = rec["betroffene"]
    print("║ ── Betroffen ──")
    if b["gesetze"]:      print(f"║   Gesetze/Normen: {', '.join(b['gesetze'])}")
    if b["institutionen"]: print(f"║   Institutionen: {', '.join(b['institutionen'])}")
    if b["gruppen"]:      print(f"║   Gruppen: {', '.join(b['gruppen'])}")
    if b["kosten"]:       print(f"║   Kosten: {b['kosten']}")
    if rec["adversarial"]:
        print("║ ── Adversarial-Einwände (gemma4) ──")
        for e in rec["adversarial"]:
            print(f"║   ⚔ {e['dimension']} → {e['empfohlene_ampel']}: {e['einwand']}")
    vh = rec.get("verhaeltnismaessigkeit", {})
    print(f"║ ▓▓ ABWÄGUNG")
    print(f"║   Verhältnismäßigkeit: {vh.get('status','')} — {vh.get('begruendung','')}")
    for al in rec.get("alternativen", []):
        print(f"║   Alternative: {al['option']} — {al['begruendung']}")
    print(f"║ ═══ FAZIT: {rec.get('fazit','')}" + ("  ‼ REVISIONSBEDÜRFTIG" if rec["revision_needed"] else ""))
    print("╚══ Quellen:", ", ".join((q or "")[:55] for q in rec["quellen"][:4]))


def _make_llms():
    """Drei Familien wie die Hauptpipeline: Erzeugung=Mistral (mit RAG faktentreu),
    Adversarial=gemma4, Critique=athena. Liefert (gen, adv, critique, host)."""
    from langchain_ollama import OllamaLLM
    from langchain_mistralai import ChatMistralAI
    from critique import create_critique_chain
    host = os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434")
    gen = ChatMistralAI(model="mistral-large-latest", api_key=os.environ["MISTRAL_API_KEY"],
                        temperature=0.0, max_tokens=3000, timeout=240)
    adv = OllamaLLM(model=os.getenv("ADVERSARIAL_MODEL", "gemma4:26b"), base_url=host,
                    timeout=900, num_ctx=16384, num_predict=900, reasoning=False, format="json")
    critique = create_critique_chain(model=os.getenv("CRITIQUE_MODEL", "athena:latest"), host=host)
    return gen, adv, critique, host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="Positions-Modus: nur dieser slug")
    ap.add_argument("--limit", type=int, default=99)
    ap.add_argument("--reform", help="Standalone: beliebiges Vorhaben als Text analysieren")
    ap.add_argument("--reform-file", help="Standalone: Vorhaben aus Datei")
    ap.add_argument("--title", default="", help="Standalone: Titel für die Ausgabe")
    ap.add_argument("--out", help="Standalone: Ergebnis als JSON hierhin schreiben")
    a = ap.parse_args()

    import serve
    vs, _ = serve._get_components("bund")
    gen, adv, critique, host = _make_llms()

    # ── Standalone-Modus ──────────────────────────────────────────────
    reform = a.reform
    if a.reform_file:
        reform = pathlib.Path(a.reform_file).read_text(encoding="utf-8")
    if reform:
        titel = a.title or reform.strip().split("\n")[0][:70]
        print(f"… Standalone-Umsetzungsanalyse: {titel}", file=sys.stderr)
        rec, _ = analyze(reform, reform.strip()[:300], vs, gen, adv, critique, host, serve,
                         heute=datetime.date.today().isoformat())
        _print_human(titel, rec)
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps({"titel": titel, **rec}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] → {a.out}", file=sys.stderr)
        return

    # ── Positions-Modus (EVIDENZ-Programm) ────────────────────────────
    data = {}
    if UM_JSON.exists():
        try:
            data = json.loads(UM_JSON.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    done = 0
    for fname, slug in POSITIONS:
        if a.only and slug != a.only:
            continue
        if done >= a.limit:
            break
        md_path = OUT_MD / fname
        if not md_path.exists():
            print(f"[skip] {slug}: {fname} fehlt", file=sys.stderr); continue
        md = md_path.read_text(encoding="utf-8")
        fm = re.search(r"(?:Entscheidungsfrage|Worum geht's)[:\s*]*\n?(.+)", md)
        query = (fm.group(1) if fm else slug).strip()[:300]
        print(f"… {slug}: Umsetzungsanalyse …", file=sys.stderr)
        try:
            rec, _ = analyze(md, query, vs, gen, adv, critique, host, serve)
        except Exception as e:
            print(f"  [fail] {slug}: {type(e).__name__}: {e}", file=sys.stderr); continue
        data[slug] = rec
        UM_JSON.parent.mkdir(parents=True, exist_ok=True)
        UM_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        flag = "‼ NICHT umsetzbar" if rec["gesamt"] == "nicht_umsetzbar" else rec["gesamt"]
        rev = " · REVISION" if rec["revision_needed"] else ""
        print(f"  [ok] {slug}: gesamt={flag}{rev} · {rec['umsetzungsdauer']} ({rec['dauer_jahre']})"
              f" · {len(rec['adversarial'])} Adversarial-Einwand/-Einwände", file=sys.stderr)
        done += 1
    print(f"[done] {done} Positionen → {UM_JSON}", file=sys.stderr)


if __name__ == "__main__":
    main()
