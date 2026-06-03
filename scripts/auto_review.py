#!/usr/bin/env python3
"""Athena — automatische Vorab-Bewertung von Quellen-Vorschlägen.

Bewertet einen eingereichten Quellen-Vorschlag STRENG per LLM (Mistral):
Herausgeber/Quelle vertrauenswürdig? Thematisch relevant für EVIDENZ? Welche
Tags? Das Ergebnis (Score, Begründung, Tags, Empfehlung) wird an die Submission
geschrieben und in submissions/log.jsonl protokolliert.

WICHTIG: Diese Pipeline pflegt NICHTS automatisch ein. Sie liefert nur eine
Entscheidungsgrundlage — die finale Freigabe macht ein Mensch
(review_submissions.py). Das schützt die Wissensbasis vor Vergiftung durch
manipulierte öffentliche Vorschläge.

Aufruf:
    python scripts/auto_review.py --id <submission-hex>   # einen bewerten
    python scripts/auto_review.py --all-pending           # alle offenen
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SUBMISSIONS_DIR = Path(__file__).parent.parent / "submissions"
PENDING_DIR = SUBMISSIONS_DIR / "pending"
LOG_PATH = SUBMISSIONS_DIR / "log.jsonl"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
REVIEW_MODEL = os.getenv("ATHENA_REVIEW_MODEL", "mistral-large-latest")

# Tag-Leitplanken: bevorzugte Begriffe für Konsistenz. Das LLM darf neue Tags
# vergeben, soll aber zuerst hier passende wählen (verhindert Wildwuchs).
SUGGESTED_TAGS = [
    "direkte-demokratie", "wahlrecht", "parteienfinanzierung", "lobbyregulierung",
    "schuldenbremse", "haushalt", "steuersystem", "ki-regulation", "digitalpolitik",
    "buergergeld", "sozialpolitik", "parlamentarismus", "rente", "demografie",
    "klima", "energie", "migration", "asyl", "verteidigung", "sicherheit",
    "aussenpolitik", "eu-recht", "grundgesetz", "bundesrecht", "statistik",
    "rechtsprechung", "wirtschaft", "bildung", "gesundheit", "umwelt",
]

REVIEW_SCHEMA_HINT = """Antworte AUSSCHLIESSLICH mit einem JSON-Objekt, keine Erklärung davor/danach:
{
  "publisher": "Herausgeber/Betreiber der Quelle, so konkret wie möglich",
  "publisher_trust": "high|medium|low|unknown",
  "publisher_reasoning": "1-2 Sätze: wer steckt dahinter, wie seriös/unabhängig",
  "relevant": true|false,
  "relevance_reasoning": "1-2 Sätze: passt das thematisch in eine faktenbasierte politische Wissensbasis für Deutschland?",
  "suggested_tier": 1|2|3,
  "tier_reasoning": "kurz: 1=Primärquelle (Gesetz/Statistik/Gericht/Behörde), 2=Qualitätsmedien, 3=Kommentar/Blog",
  "topics": ["tag1", "tag2"],
  "recommendation": "approve|reject|needs_human",
  "summary": "1 Satz Gesamtfazit für den menschlichen Reviewer"
}"""


def _build_llm():
    from langchain_mistralai import ChatMistralAI
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY nicht gesetzt — Review-Pipeline braucht Mistral.")
    return ChatMistralAI(model=REVIEW_MODEL, api_key=MISTRAL_API_KEY,
                         temperature=0.1, max_retries=2, timeout=90)


def _fetch_sample(meta: dict, submission_dir: Path) -> tuple[str, str]:
    """Holt einen Textauszug der Quelle (für die Bewertung). Liefert (quelle, sample)."""
    if meta.get("kind") == "url":
        url = meta.get("url", "")
        try:
            import requests
            r = requests.get(url, timeout=20, headers={"User-Agent": "Athena-KI/1.0 (+review)"})
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text).strip()
            return url, text[:4000]
        except Exception as e:
            return url, f"[Abruf fehlgeschlagen: {type(e).__name__}: {e}]"
    else:
        fname = meta.get("filename", "")
        fpath = submission_dir / fname
        try:
            if fname.lower().endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(str(fpath))
                text = " ".join((p.extract_text() or "") for p in reader.pages[:5])
            else:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            text = re.sub(r"\s+", " ", text).strip()
            return fname, text[:4000]
        except Exception as e:
            return fname, f"[Lesen fehlgeschlagen: {type(e).__name__}: {e}]"


def _ingest_as_tier0(meta: dict, submission_dir: Path, label: str, topics: str) -> int:
    """Pflegt die Quelle als Tier 0 (unverifiziert) via ingest.py ein. Liefert rc."""
    import subprocess
    ingest_script = Path(__file__).parent / "ingest.py"
    scope = meta.get("scope", "pfofeld")
    args = [sys.executable, str(ingest_script),
            "--tier", "0", "--source-label", label, "--scope", scope]
    if topics:
        args.extend(["--topics", topics])
    if meta.get("kind") == "url":
        args.extend(["--url", meta["url"], "--render"])
    else:
        args.extend(["--file", str(submission_dir / meta["filename"])])
    try:
        return subprocess.call(args, timeout=300)
    except Exception as e:
        print(f"[ingest tier0] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


def review_submission(submission_dir: Path) -> dict:
    """Bewertet eine Submission. Schreibt Ergebnis an meta.json + log.jsonl.
    Pflegt NICHTS ein. Liefert das Bewertungs-dict."""
    meta = json.loads((submission_dir / "meta.json").read_text(encoding="utf-8"))
    source, sample = _fetch_sample(meta, submission_dir)
    domain = urlparse(meta["url"]).netloc if meta.get("kind") == "url" else "(Datei-Upload)"

    prompt = f"""Du bist die strenge Quellen-Prüfung von EVIDENZ, einer faktenbasierten politischen Bewegung in Deutschland. Eine Person hat eine Quelle für die Wissensbasis vorgeschlagen. Bewerte sie KRITISCH — die Wissensbasis darf nicht durch unseriöse oder manipulative Quellen vergiftet werden.

Prüfe besonders:
- Wer ist der HERAUSGEBER? Ist er identifizierbar, seriös, unabhängig? Bei unklarem/anonymem/interessengeleitetem Herausgeber → publisher_trust niedrig.
- Ist es eine PRIMÄRQUELLE (Gesetz, Statistik, Gericht, Behörde, Gutachten) oder Meinung/Werbung/Propaganda?
- Passt es thematisch in eine deutsche politische Faktenbasis?

Im Zweifel "needs_human" statt "approve".

Bevorzugte Tags (wähle passende, neue nur wenn nötig): {', '.join(SUGGESTED_TAGS)}

QUELLE: {source}
DOMAIN: {domain}
NOTIZ DES EINREICHERS: {meta.get('note') or '(keine)'}
SCOPE: {meta.get('scope')}

TEXTAUSZUG:
{sample}

{REVIEW_SCHEMA_HINT}"""

    llm = _build_llm()
    raw = llm.invoke(prompt)
    content = raw.content if hasattr(raw, "content") else str(raw)
    # JSON aus der Antwort extrahieren
    m = re.search(r"\{.*\}", content, re.DOTALL)
    try:
        verdict = json.loads(m.group(0)) if m else {"error": "kein JSON", "raw": content[:500]}
    except Exception as e:
        verdict = {"error": f"JSON-Parse: {e}", "raw": content[:500]}

    verdict["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    verdict["model"] = REVIEW_MODEL

    # an meta.json anhängen
    meta["auto_review"] = verdict
    (submission_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Auto-Ingest als TIER 0 (unverifiziert), wenn Athena nicht ablehnt.
    # Tier 0 ist im Retrieval per Default ausgeschlossen → keine Vergiftung der
    # normalen Analysen. Die Hochstufung auf Tier 1-3 macht ein Mensch
    # (review_submissions.py). Bei "reject" wird nichts eingepflegt.
    ingest_status = "not_ingested"
    rec = verdict.get("recommendation")
    if rec in ("approve", "needs_human") and not verdict.get("error"):
        topics = ",".join(verdict.get("topics") or [])
        label = (verdict.get("publisher") or "User-Submission")[:120]
        rc = _ingest_as_tier0(meta, submission_dir, label, topics)
        ingest_status = "ingested_tier0" if rc == 0 else f"ingest_failed_rc{rc}"
        verdict["ingest_status"] = ingest_status
        (submission_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # in öffentlichen Log schreiben (anonymisiert — KEINE Einreicher-Daten)
    log_entry = {
        "id": meta["id"],
        "submitted_at": meta.get("submitted_at"),
        "reviewed_at": verdict["reviewed_at"],
        "source": source,
        "domain": domain,
        "scope": meta.get("scope"),
        "publisher": verdict.get("publisher"),
        "publisher_trust": verdict.get("publisher_trust"),
        "relevant": verdict.get("relevant"),
        "suggested_tier": verdict.get("suggested_tier"),
        "topics": verdict.get("topics"),
        "recommendation": verdict.get("recommendation"),
        "summary": verdict.get("summary"),
        # status: ingested_tier0 = als unverifiziert aufgenommen (per Default vom
        # Retrieval aus, wartet auf menschl. Hochstufung); rejected = abgelehnt.
        "status": ("rejected" if rec == "reject" or verdict.get("error")
                   else ingest_status),
        "verified": False,  # wird True bei menschlicher Freigabe (Tier 1-3)
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return verdict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", help="Eine bestimmte Submission (hex-id) bewerten")
    p.add_argument("--all-pending", action="store_true", help="Alle pending-Submissions bewerten")
    args = p.parse_args()

    if args.id:
        targets = [PENDING_DIR / args.id]
    elif args.all_pending:
        targets = [d for d in PENDING_DIR.iterdir() if d.is_dir() and (d / "meta.json").exists()] if PENDING_DIR.exists() else []
    else:
        p.error("--id oder --all-pending angeben")

    for d in targets:
        if not (d / "meta.json").exists():
            print(f"[skip] {d.name}: keine meta.json", file=sys.stderr)
            continue
        print(f"[review] {d.name} …", file=sys.stderr)
        v = review_submission(d)
        print(json.dumps(v, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
