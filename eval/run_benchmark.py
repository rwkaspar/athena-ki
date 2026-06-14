#!/usr/bin/env python3
"""Athena-Benchmark v0 — Vergleich Athena (RAG) vs. Standard-LLMs.

Läuft das Gold-Set (eval/gold_set_v0.yaml) gegen mehrere Routes:
  - athena   — Athenas /chat (RAG + Quellenangabe)
  - mistral  — Mistral-Large ohne RAG
  - openai   — ChatGPT (gpt-4o-mini default) ohne RAG
  - claude   — Claude (claude-sonnet-4-6 default) ohne RAG

Bewertet pro Antwort:
  - fact_hit           — enthält Antwort mindestens eine der 'accept'-Varianten?
                         Beachtet 'accept_mode' (any|all|at_least_three).
  - hallucinated_urls  — Anzahl URLs in der Antwort, die nicht aufrufbar sind
                         (HEAD 4xx/5xx, Timeout, DNS-Fail).
  - sources_cited      — Anzahl der mitgelieferten Quellen (nur Athena).
  - declined_uncertain — hat das Modell "weiß ich nicht" / "keine Aussage
                         möglich" geantwortet? (Heuristik per Substring.)

Aufruf
------
    python eval/run_benchmark.py --routes athena,openai,claude
    python eval/run_benchmark.py --routes athena --limit 5   # Smoke-Test

Umgebungsvariablen:
    ATHENA_API_BASE   — z. B. http://aitest:8765  (Default: localhost:8765)
    MISTRAL_API_KEY   — für route 'mistral'
    OPENAI_API_KEY    — für route 'openai'
    OPENAI_MODEL      — Default: gpt-4o-mini
    ANTHROPIC_API_KEY — für route 'claude'
    ANTHROPIC_MODEL   — Default: claude-sonnet-4-6

Ergebnisse landen in eval/results/<ts>_<route>.json (per-Frage + Summary).
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "results"
GOLD_DEFAULT = EVAL_DIR / "gold_set_v0.yaml"

# Heuristik: Antworten, die ehrlich auf dünne Faktenlage zeigen.
DECLINE_PATTERNS = [
    "keine zuverlässige", "keine belastbare", "nicht sicher", "weiß ich nicht",
    "kann ich nicht beantworten", "keine ausreichenden quellen", "keine konkreten zahlen",
    "ich habe dazu keine quellen", "lässt sich nicht eindeutig", "lässt sich nicht genau",
]

URL_RE = re.compile(r"https?://[^\s)\]<>\"]+", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

def route_athena(question: str) -> dict:
    """Ruft Athenas /chat-Endpoint (NDJSON-Stream) auf — RAG + Quellen."""
    import requests
    base = os.getenv("ATHENA_API_BASE", "http://localhost:8765").rstrip("/")
    url = base + "/chat"
    payload = {"message": question, "scope": "bund", "include_unverified": False}
    text, sources = "", []
    with requests.post(url, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "sources":
                sources = ev.get("sources") or []
            elif ev.get("type") == "token":
                text += ev.get("text", "")
    return {"answer": text.strip(), "sources": sources}


def route_mistral(question: str) -> dict:
    """Plain Mistral via langchain_mistralai — kein RAG, kein Quellen-Boost."""
    from langchain_mistralai import ChatMistralAI
    model = os.getenv("ATHENA_REVIEW_MODEL", "mistral-large-latest")
    llm = ChatMistralAI(model=model, api_key=os.getenv("MISTRAL_API_KEY", ""),
                        temperature=0.1, max_retries=2, timeout=120)
    r = llm.invoke(question)
    return {"answer": (r.content if hasattr(r, "content") else str(r)).strip(),
            "sources": []}


def route_openai(question: str) -> dict:
    """Plain ChatGPT (gpt-4o-mini default) — kein RAG."""
    import requests
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','')}"},
        json={"model": model, "temperature": 0.1,
              "messages": [{"role": "user", "content": question}]},
        timeout=180,
    )
    r.raise_for_status()
    j = r.json()
    return {"answer": j["choices"][0]["message"]["content"].strip(), "sources": []}


def route_claude(question: str) -> dict:
    """Plain Claude (claude-sonnet-4-6 default) — kein RAG."""
    import requests
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"model": model, "max_tokens": 2048, "temperature": 0.1,
              "messages": [{"role": "user", "content": question}]},
        timeout=180,
    )
    r.raise_for_status()
    j = r.json()
    # Anthropic-API liefert content als Liste von Blöcken
    parts = [b.get("text", "") for b in j.get("content", []) if b.get("type") == "text"]
    return {"answer": "".join(parts).strip(), "sources": []}


ROUTES = {
    "athena": route_athena,
    "mistral": route_mistral,
    "openai": route_openai,
    "claude": route_claude,
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def fact_hit(answer: str, accept: list[str], mode: str = "any") -> bool:
    """Substring-Match auf 'accept'. mode: any | all | at_least_three."""
    if not answer or not accept:
        return False
    low = answer.lower()
    hits = sum(1 for a in accept if a.lower() in low)
    if mode == "all":
        return hits == len(accept)
    if mode == "at_least_three":
        return hits >= 3
    return hits >= 1


def declined(answer: str) -> bool:
    low = (answer or "").lower()
    return any(p in low for p in DECLINE_PATTERNS)


def check_urls(answer: str, timeout: float = 4.0) -> dict:
    """Listet URLs in der Antwort und zählt, wie viele nicht erreichbar sind.
    HEAD-Request, kurzer Timeout — wir wollen nur grobes Halluzinations-Signal."""
    import requests
    urls = URL_RE.findall(answer or "")
    bad = 0
    for u in urls:
        try:
            r = requests.head(u, allow_redirects=True, timeout=timeout)
            if r.status_code >= 400:
                # manche Server lehnen HEAD ab → mit GET nochmal probieren
                rg = requests.get(u, timeout=timeout, stream=True)
                if rg.status_code >= 400:
                    bad += 1
        except Exception:
            bad += 1
    return {"urls": urls, "url_count": len(urls), "bad_url_count": bad}


def grade(question: dict, answer: str, sources: list[str]) -> dict:
    accept = question.get("accept") or []
    mode = question.get("accept_mode", "any")
    hit = fact_hit(answer, accept, mode)
    url_info = check_urls(answer)
    return {
        "fact_hit": int(hit),
        "declined_uncertain": int(declined(answer)),
        "sources_cited": len(sources or []),
        "answer_chars": len(answer or ""),
        "url_count": url_info["url_count"],
        "bad_url_count": url_info["bad_url_count"],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run_route(name: str, fn, gold: list[dict], limit: int | None) -> dict:
    items = gold if not limit else gold[:limit]
    rows = []
    print(f"\n=== route: {name} ({len(items)} Fragen) ===", file=sys.stderr)
    for i, q in enumerate(items, 1):
        t0 = time.time()
        try:
            r = fn(q["question"])
            err = None
        except Exception as e:
            r = {"answer": "", "sources": []}
            err = f"{type(e).__name__}: {e}"
        dt = round(time.time() - t0, 2)
        g = grade(q, r["answer"], r.get("sources") or [])
        row = {
            "id": q["id"], "topic": q.get("topic"),
            "question": q["question"], "gold_answer": q.get("gold_answer"),
            "answer": r["answer"], "sources": r.get("sources") or [],
            "latency_s": dt, "error": err,
            **g,
        }
        rows.append(row)
        marker = "✓" if g["fact_hit"] else ("◌" if g["declined_uncertain"] else "✗")
        print(f"  [{i:2d}/{len(items)}] {marker} {q['id']:20s} {dt:5.1f}s  hit={g['fact_hit']} sources={g['sources_cited']} bad_urls={g['bad_url_count']}",
              file=sys.stderr)
    summary = {
        "route": name,
        "total": len(rows),
        "fact_hits": sum(r["fact_hit"] for r in rows),
        "declined": sum(r["declined_uncertain"] for r in rows),
        "errors": sum(1 for r in rows if r["error"]),
        "avg_sources": round(sum(r["sources_cited"] for r in rows) / max(1, len(rows)), 2),
        "avg_bad_urls": round(sum(r["bad_url_count"] for r in rows) / max(1, len(rows)), 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / max(1, len(rows)), 2),
    }
    print(f"  → {summary['fact_hits']}/{summary['total']} fact_hits, "
          f"{summary['declined']} declined, {summary['errors']} errors, "
          f"avg_sources={summary['avg_sources']}, avg_bad_urls={summary['avg_bad_urls']}",
          file=sys.stderr)
    return {"summary": summary, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-set", default=str(GOLD_DEFAULT))
    p.add_argument("--routes", default="athena",
                   help="kommagetrennt: athena,mistral,openai,claude")
    p.add_argument("--limit", type=int, default=None,
                   help="Nur erste N Fragen — für Smoke-Tests.")
    p.add_argument("--out", default=str(RESULTS_DIR))
    args = p.parse_args()

    gold = yaml.safe_load(Path(args.gold_set).read_text(encoding="utf-8"))
    questions = gold["questions"]
    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    all_summaries = []
    for r in routes:
        if r not in ROUTES:
            print(f"[skip] unbekannte route: {r}", file=sys.stderr)
            continue
        result = run_route(r, ROUTES[r], questions, args.limit)
        out_path = out_dir / f"{ts}_{r}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"  → {out_path}", file=sys.stderr)
        all_summaries.append(result["summary"])

    # Gesamt-Vergleich auf stderr
    if len(all_summaries) > 1:
        print("\n=== Vergleich ===", file=sys.stderr)
        for s in all_summaries:
            print(f"  {s['route']:10s} {s['fact_hits']:3d}/{s['total']:3d} hits  "
                  f"declined={s['declined']:2d}  bad_urls={s['avg_bad_urls']:.2f}  "
                  f"sources={s['avg_sources']:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
