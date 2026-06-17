#!/usr/bin/env python3
"""Einmalig: vollständige Critique (num_predict jetzt 4096) + Critique-Verdikt für einen
bereits erzeugten pipeline_demo_*.json nachziehen — Retrieval/Position/Adversarial bleiben
unangetastet. Aufruf: OLLAMA_HOST=… CRITIQUE_MODEL=athena:latest python scripts/_backfill_critique.py docs/pipeline_demo_ki.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

path = sys.argv[1]
d = json.load(open(path, encoding="utf-8"))
q, scope = d["question"], d.get("scope", "bund")

import serve
from retrieval import tier_aware_retrieve
from critique import create_critique_chain
from pipeline_demo import critique_verdict

host = os.environ.get("OLLAMA_HOST", "http://100.101.225.56:11434")
print("… Retrieval", file=sys.stderr)
vs, _ = serve._get_components(scope)
docs = tier_aware_retrieve(vs, q, k=serve.RETRIEVER_K, fetch_k=50, sim_floor=0.45, max_k=15)
print(f"… Critique (vollständig, {len(docs)} docs)", file=sys.stderr)
crit = create_critique_chain()(q, docs, d["answer"])
print(f"… Critique {len(crit)} Zeichen, endet auf: {crit[-50:]!r}", file=sys.stderr)
print("… Verdikt", file=sys.stderr)
verd = critique_verdict(crit, host)
print(f"… Verdikt: {verd}", file=sys.stderr)

d["critique"] = crit
d["critique_verdict"] = verd
json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
print(f"[ok] → {path}", file=sys.stderr)
