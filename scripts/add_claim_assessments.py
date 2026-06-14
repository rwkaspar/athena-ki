#!/usr/bin/env python3
"""Reichert Themen-Dossiers um PRO-BEHAUPTUNG-Verdikte an (claim_assessments).

Für jede Partei werden ihre key_claims einzeln eingeordnet — auf Basis der BEREITS
vorhandenen Athena-Analyse (summary/empirical_check/verdict). Es werden KEINE neuen
Fakten erfunden; das Skript schlüsselt nur die schon getroffene Bewertung pro
Behauptung auf, damit Leser sehen: welcher Teil gestützt, welcher nicht, warum.

Aufruf:
    MISTRAL_API_KEY=… python scripts/add_claim_assessments.py eval/topic_dossiers/wahlrecht.json
    … (mehrere Dateien möglich)
"""
import json, os, re, sys, time
KEY = os.environ.get("MISTRAL_API_KEY", "")
if not KEY:
    sys.exit("MISTRAL_API_KEY fehlt")
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatMistralAI(model="mistral-large-latest", api_key=KEY, temperature=0.1,
                    max_tokens=2000, timeout=120)

VERDICTS = ("gestützt", "teilweise gestützt", "teilweise widerlegt", "widerlegt", "offen")
SYS = f"""Du ordnest die EINZELNEN Kernbehauptungen einer Partei ein — ausschließlich
auf Basis der bereits vorliegenden Athena-Analyse. Erfinde KEINE neuen Fakten und
keine neuen Quellen; leite jedes Einzel-Verdikt aus der gelieferten Zusammenfassung
und empirischen Prüfung ab. Bleib knapp und neutral, gleiche Strenge für alle Parteien.

Antworte mit GENAU EINEM JSON-Array, ein Objekt je gelieferter Behauptung:
[{{"claim": "<die Behauptung wortgleich>", "verdict": "<{'|'.join(VERDICTS)}>", "reason": "<ein knapper Satz: warum dieses Verdikt>"}}]
Nur JSON, keine Erklärung, keine Markdown-Fences."""


def extract_json(t):
    t = re.sub(r"^```(json)?|```$", "", t.strip(), flags=re.M).strip()
    i, j = t.find("["), t.rfind("]")
    return t[i:j + 1] if i >= 0 else t


def assess_party(pos):
    claims = pos.get("key_claims") or []
    if not claims:
        return []
    a = pos.get("athena_analysis") or {}
    user = (f"Partei: {pos.get('party')}\n"
            f"Gesamt-Verdikt: {a.get('verdict')}\n"
            f"Zusammenfassung: {a.get('summary')}\n"
            f"Empirische Prüfung: {a.get('empirical_check')}\n\n"
            f"Behauptungen (einzeln einordnen):\n" + "\n".join(f"- {c}" for c in claims))
    # Retry mit Backoff gegen Mistral-Rate-Limit (429).
    for attempt in range(5):
        try:
            resp = llm.invoke([SystemMessage(content=SYS), HumanMessage(content=user)])
            break
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                time.sleep(8 * (attempt + 1))
                continue
            raise
    arr = json.loads(extract_json(resp.content or ""))
    out = []
    for c in arr:
        v = (c.get("verdict") or "offen").strip().lower()
        if v not in VERDICTS:
            v = "offen"
        out.append({"claim": c.get("claim", ""), "verdict": v, "reason": c.get("reason", "")})
    return out


for path in sys.argv[1:]:
    d = json.loads(open(path, encoding="utf-8").read())
    for pos in d.get("party_positions", []):
        try:
            ca = assess_party(pos)
            time.sleep(2)  # Throttle gegen Rate-Limit
            pos.setdefault("athena_analysis", {})["claim_assessments"] = ca
            print(f"  [ok] {d.get('slug')} · {pos.get('party')}: {len(ca)} Behauptungen", file=sys.stderr)
        except Exception as e:
            print(f"  [!] {d.get('slug')} · {pos.get('party')}: {type(e).__name__}: {e}", file=sys.stderr)
    open(path, "w", encoding="utf-8").write(json.dumps(d, ensure_ascii=False, indent=2))
    print(f"[geschrieben] {path}", file=sys.stderr)
