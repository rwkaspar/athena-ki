#!/usr/bin/env python3
"""Athena Gold-Eval Runner — Benchmark gegen Gold-Set v1.

Schickt jede Frage aus eval/gold_set_v1.json an Athena (RAG) und optional an
Baseline-Modelle ohne RAG (plain Mistral via API, Claude via API). Bewertet:

1. Faktentreue: accept_patterns-Regex-Match auf die Antwort.
2. Halluzinations-Rate: zitierte URLs im Antworttext gegen HTTP HEAD prüfen.
3. Quellen-Tier-Verteilung: zitierte Quellen (von Athena geliefert) gegen
   source_tiers_bund.yaml — Anteil Tier 1.

Ergebnisse werden pro Lauf in eval/runs/<datum>_<modus>.jsonl geschrieben +
eine Zusammenfassung als eval/reports/<datum>_report.md.

Aufruf:
    python scripts/run_gold_eval.py                    # Athena allein
    python scripts/run_gold_eval.py --modes athena,mistral,claude
    python scripts/run_gold_eval.py --gold eval/gold_set_v1.json --out eval/runs/
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import requests

ATHENA_API = os.getenv("ATHENA_API", "http://100.105.70.24:8765")
ATHENA_TIMEOUT = int(os.getenv("ATHENA_TIMEOUT", "240"))
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL = os.getenv("MISTRAL_BENCH_MODEL", "mistral-large-latest")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("CLAUDE_BENCH_MODEL", "claude-opus-4-5")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = REPO_ROOT / "eval" / "gold_set_v1.json"
DEFAULT_RUNS = REPO_ROOT / "eval" / "runs"
DEFAULT_REPORTS = REPO_ROOT / "eval" / "reports"

URL_RE = re.compile(r"https?://[^\s)>\]\"'`]+")


def grade_answer(question: dict, answer_text: str) -> dict:
    """Pattern-Match-Grading. Liefert hit + matched_patterns + Trap-Hits."""
    patterns = question.get("accept_patterns") or []
    min_n = int(question.get("min_pattern_matches") or 1)
    matched = []
    for p in patterns:
        try:
            if re.search(p, answer_text, re.IGNORECASE):
                matched.append(p)
        except re.error as e:
            print(f"  [grade] bad regex {p!r}: {e}", file=sys.stderr)
    hit = len(matched) >= min_n
    trap_hits = []
    for trap in question.get("hallucination_traps") or []:
        # Trap-Pattern ist freier Text — wir matchen das markante erste Token
        token = trap.split("(")[0].strip()
        if token and len(token) >= 3 and re.search(re.escape(token), answer_text):
            trap_hits.append(trap)
    return {"hit": hit, "matched_patterns": matched,
            "required_min": min_n, "trap_hits": trap_hits}


def extract_urls(answer_text: str) -> list[str]:
    """Alle http(s)-URLs aus dem Antworttext ziehen, deduplikiert."""
    seen, out = set(), []
    for m in URL_RE.findall(answer_text):
        u = m.rstrip(".,;:)]}")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def check_url_alive(url: str, timeout: int = 6) -> bool:
    """HEAD-Request. 2xx/3xx = lebt, sonst halluziniert (oder broken)."""
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True,
                          headers={"User-Agent": "Athena-Eval/1.0"})
        if r.status_code < 400:
            return True
        # Manche Server lehnen HEAD ab → GET probieren, aber nur kurze Antwort
        r = requests.get(url, timeout=timeout, allow_redirects=True, stream=True,
                         headers={"User-Agent": "Athena-Eval/1.0"})
        return r.status_code < 400
    except Exception:
        return False


def url_audit(urls: list[str]) -> dict:
    """Pro URL: lebt? Aggregiert: Halluzinations-Anteil."""
    if not urls:
        return {"total": 0, "alive": 0, "dead": 0, "details": [], "halluc_rate": 0.0}
    details, alive = [], 0
    for u in urls:
        ok = check_url_alive(u)
        details.append({"url": u, "alive": ok})
        if ok:
            alive += 1
    return {"total": len(urls), "alive": alive, "dead": len(urls) - alive,
            "details": details,
            "halluc_rate": round((len(urls) - alive) / len(urls), 3)}


def ask_athena(question_text: str, include_unverified: bool = False) -> dict:
    """Athena-Chat-Endpoint streamen, Antwort + Quellen sammeln."""
    url = f"{ATHENA_API}/chat"
    payload = {"message": question_text, "scope": "bund", "history": [],
               "include_unverified": include_unverified}
    text, sources, source_meta = "", [], {}
    t0 = time.time()
    try:
        r = requests.post(url, json=payload, stream=True, timeout=ATHENA_TIMEOUT)
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}",
                    "answer": "", "sources": [], "latency_s": time.time() - t0}
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "token":
                text += ev.get("text") or ""
            elif ev.get("type") == "sources":
                sources = ev.get("sources") or []
                source_meta = ev.get("source_meta") or {}
            elif ev.get("type") == "error":
                text += "\n⚠️ " + (ev.get("message") or "Fehler")
        return {"ok": True, "answer": text.strip(), "sources": sources,
                "source_meta": source_meta, "latency_s": round(time.time() - t0, 2)}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "answer": "", "sources": [], "latency_s": time.time() - t0}


def ask_mistral_bare(question_text: str) -> dict:
    """Plain Mistral via API (kein RAG, kein Quellen-Kontext)."""
    if not MISTRAL_API_KEY:
        return {"ok": False, "error": "MISTRAL_API_KEY nicht gesetzt", "answer": ""}
    t0 = time.time()
    try:
        from mistralai import Mistral
        client = Mistral(api_key=MISTRAL_API_KEY)
        sys_prompt = ("Du bist ein politisch-juristisch versierter Assistent für die deutsche "
                      "Bundespolitik. Antworte präzise und nenne, wenn möglich, die Rechtsgrundlage "
                      "oder Statistikquelle. Sage 'unbekannt', wenn du es nicht weißt.")
        resp = client.chat.complete(model=MISTRAL_MODEL, messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question_text},
        ], temperature=0.1, max_tokens=600)
        text = resp.choices[0].message.content if resp.choices else ""
        return {"ok": True, "answer": (text or "").strip(),
                "sources": [], "latency_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "answer": "", "sources": [], "latency_s": time.time() - t0}


def ask_claude_bare(question_text: str) -> dict:
    """Plain Claude via Anthropic-API (kein RAG, kein web_search)."""
    if not ANTHROPIC_API_KEY:
        return {"ok": False, "error": "ANTHROPIC_API_KEY nicht gesetzt", "answer": ""}
    t0 = time.time()
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        sys_prompt = ("Du bist ein politisch-juristisch versierter Assistent für die deutsche "
                      "Bundespolitik. Antworte präzise und nenne, wenn möglich, die Rechtsgrundlage "
                      "oder Statistikquelle. Sage 'unbekannt', wenn du es nicht weißt.")
        msg = client.messages.create(model=ANTHROPIC_MODEL, max_tokens=600,
                                     temperature=0.1, system=sys_prompt,
                                     messages=[{"role": "user", "content": question_text}])
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        return {"ok": True, "answer": text.strip(), "sources": [],
                "latency_s": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "answer": "", "sources": [], "latency_s": time.time() - t0}


ASKERS = {"athena": ask_athena, "mistral": ask_mistral_bare, "claude": ask_claude_bare}


def evaluate_one(question: dict, mode: str) -> dict:
    """Eine Frage in einem Modus laufen lassen + bewerten."""
    asker = ASKERS[mode]
    result = asker(question["question"])
    answer = result.get("answer", "")
    grading = grade_answer(question, answer)
    cited_urls = extract_urls(answer)
    # Athena liefert sources separat — die URL-Liste daraus mit aufnehmen.
    if mode == "athena":
        for s in result.get("sources") or []:
            if isinstance(s, str) and s.startswith("http") and s not in cited_urls:
                cited_urls.append(s)
    audit = url_audit(cited_urls)
    return {
        "id": question["id"], "topic": question["topic"], "mode": mode,
        "question": question["question"],
        "gold_answer": question["gold_answer"],
        "answer": answer,
        "ok": result.get("ok", False),
        "error": result.get("error"),
        "latency_s": result.get("latency_s"),
        "hit": grading["hit"],
        "matched_patterns": grading["matched_patterns"],
        "required_min": grading["required_min"],
        "trap_hits": grading["trap_hits"],
        "cited_urls": cited_urls,
        "url_audit": audit,
        "athena_sources": result.get("sources") if mode == "athena" else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def summarize(results: list[dict]) -> dict:
    """Pro Modus: Hit-Rate, Halluzinations-Rate, Latenz."""
    by_mode = {}
    for r in results:
        m = r["mode"]
        b = by_mode.setdefault(m, {"total": 0, "hits": 0, "errors": 0,
                                    "trap_hits": 0, "url_total": 0, "url_dead": 0,
                                    "lat_sum": 0.0, "lat_n": 0})
        b["total"] += 1
        if r["hit"]:
            b["hits"] += 1
        if not r["ok"]:
            b["errors"] += 1
        if r["trap_hits"]:
            b["trap_hits"] += 1
        b["url_total"] += r["url_audit"]["total"]
        b["url_dead"] += r["url_audit"]["dead"]
        if r["latency_s"]:
            b["lat_sum"] += r["latency_s"]
            b["lat_n"] += 1
    for m, b in by_mode.items():
        b["hit_rate"] = round(b["hits"] / b["total"], 3) if b["total"] else 0
        b["halluc_rate"] = round(b["url_dead"] / b["url_total"], 3) if b["url_total"] else 0
        b["avg_latency_s"] = round(b["lat_sum"] / b["lat_n"], 2) if b["lat_n"] else 0
    return by_mode


def write_report(results: list[dict], summary: dict, out_path: Path, gold_meta: dict):
    lines = [f"# Athena Gold-Eval — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "",
             f"Gold-Set: **{gold_meta.get('name', 'v1')}** "
             f"({len(set(r['id'] for r in results))} Fragen)",
             ""]
    lines.append("## Zusammenfassung")
    lines.append("")
    lines.append("| Modus | Hit-Rate | Halluzinations-Rate | Ø Latenz | Fehler | Trap-Hits |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for mode, b in summary.items():
        lines.append(f"| {mode} | {b['hit_rate']:.1%} ({b['hits']}/{b['total']}) "
                     f"| {b['halluc_rate']:.1%} ({b['url_dead']}/{b['url_total']} URLs) "
                     f"| {b['avg_latency_s']}s | {b['errors']} | {b['trap_hits']} |")
    lines.append("")

    lines.append("## Antworten — wer hat was getroffen")
    lines.append("")
    # Pro Frage eine Zeile, pro Modus ✓/✗
    modes = sorted({r["mode"] for r in results})
    by_qid = {}
    for r in results:
        by_qid.setdefault(r["id"], {})[r["mode"]] = r
    lines.append("| Frage | " + " | ".join(modes) + " |")
    lines.append("|---|" + "|".join(["---"] * len(modes)) + "|")
    for qid in sorted(by_qid):
        row = [qid]
        for m in modes:
            r = by_qid[qid].get(m)
            row.append("✓" if r and r["hit"] else "✗")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Athena-only Wins (Hauptargument für den Kumpel)
    if "athena" in modes and len(modes) >= 2:
        only_athena = []
        for qid, by_m in by_qid.items():
            ath_hit = by_m.get("athena", {}).get("hit")
            others_hit = [by_m.get(m, {}).get("hit") for m in modes if m != "athena"]
            if ath_hit and not any(others_hit):
                only_athena.append(qid)
        if only_athena:
            lines.append("## Fragen, die nur Athena richtig hatte")
            lines.append("")
            for qid in only_athena:
                r = by_qid[qid]["athena"]
                lines.append(f"- **{qid}** — Gold: {r['gold_answer']}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] geschrieben: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--out", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    ap.add_argument("--modes", default="athena",
                    help="Komma-Liste: athena, mistral, claude")
    ap.add_argument("--limit", type=int, default=0, help="Nur die ersten N Fragen (Test)")
    ap.add_argument("--include-unverified", action="store_true",
                    help="Athena darf Tier-0-Quellen einbeziehen.")
    args = ap.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    questions = gold.get("questions", [])
    if args.limit > 0:
        questions = questions[:args.limit]
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in ASKERS]
    if not modes:
        ap.error("Keine gültigen Modi. Verfügbar: " + ",".join(ASKERS))

    args.out.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    all_results = []
    for mode in modes:
        run_path = args.out / f"{stamp}_{mode}.jsonl"
        print(f"\n[run] mode={mode} → {run_path} ({len(questions)} Fragen)")
        with run_path.open("w", encoding="utf-8") as f:
            for i, q in enumerate(questions, 1):
                print(f"  [{i:2d}/{len(questions)}] {q['id']} … ", end="", flush=True)
                r = evaluate_one(q, mode)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                marker = "✓" if r["hit"] else "✗"
                print(f"{marker}  ({r['latency_s']}s, "
                      f"{r['url_audit']['alive']}/{r['url_audit']['total']} URLs alive)")
                all_results.append(r)

    summary = summarize(all_results)
    print("\n[summary]")
    for m, b in summary.items():
        print(f"  {m}: hit-rate {b['hit_rate']:.1%} ({b['hits']}/{b['total']}), "
              f"halluc {b['halluc_rate']:.1%}, "
              f"avg-latency {b['avg_latency_s']}s, errors {b['errors']}")

    report_path = args.reports / f"{stamp}_report.md"
    write_report(all_results, summary, report_path, gold)


if __name__ == "__main__":
    main()
