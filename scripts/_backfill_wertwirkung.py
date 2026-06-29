#!/usr/bin/env python3
"""Einmalig: wertwirkung (§§1-7) in die Optionen einer bestehenden pipeline_demo_*.json
nachrüsten — ein LLM-Call pro Option. Aufruf: python scripts/_backfill_wertwirkung.py docs/pipeline_demo_ki.json"""
import json, re, sys, os
from langchain_ollama import OllamaLLM

path = sys.argv[1]
canon = open(os.path.join(os.path.dirname(__file__), "..", "config", "wertekanon.md"), encoding="utf-8").read()
d = json.load(open(path, encoding="utf-8"))
opts = d.get("analysis", {}).get("optionen") or []
llm = OllamaLLM(model=os.getenv("WW_MODEL", "athena:latest"),
                base_url=os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434"),
                timeout=600, num_ctx=8192, num_predict=800, reasoning=False, format="json")
valid = {f"§{i}" for i in range(1, 8)}
PROMPT = ("Bewerte, wie diese Handlungsoption auf die §§1-7 des EVIDENZ-Wertekanons wirkt. "
          "Pro § ein Wert -100..+100 (+100 stützt stark, 0 neutral, -100 belastet stark). Keine Empfehlung. "
          'Antworte NUR als JSON: {"wertwirkung":[{"paragraph":"§1","intensitaet":<int>,"begruendung":"<1 Satz>"}, ... §1-§7]}\n\n'
          "WERTEKANON:\n" + canon + "\n\nOPTION:\n{opt}")
for o in opts:
    txt = f"{o.get('titel','')}\n{o.get('beschreibung','')}\nTrade-offs: {'; '.join(o.get('trade_offs') or [])}"
    try:
        raw = llm.invoke(PROMPT.replace("{opt}", txt))
        j = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        ww = [{"paragraph": w["paragraph"].strip(),
               "intensitaet": max(-100, min(100, int(round(float(w.get("intensitaet", 0)))))),
               "begruendung": (w.get("begruendung") or "").strip()[:200]}
              for w in j.get("wertwirkung", []) if (w.get("paragraph") or "").strip() in valid]
        o["wertwirkung"] = sorted(ww, key=lambda x: x["paragraph"])
        print(f"  {o.get('titel','')[:45]}: {len(ww)} Werte", file=sys.stderr)
    except Exception as e:
        print(f"  FAIL {o.get('titel','')[:40]}: {type(e).__name__}: {e}", file=sys.stderr)
json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
print("[ok] geschrieben", file=sys.stderr)
