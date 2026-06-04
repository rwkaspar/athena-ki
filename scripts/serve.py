"""Athena Chat-API.

Kleiner FastAPI-Wrapper um die Pipeline aus query.py, gedacht für den
NeoTactIQ-Frontend-Chat. Nur Quick-Chat (Athena + RAG, kein Verify/Critique),
weil längere Pipeline-Stufen für interaktiven Chat zu langsam sind.

Wichtig: /chat ist Streaming (NDJSON pro Zeile). Cloudflare's Free-Tier hat
~100s Timeout für nicht-streamende Antworten, Athena braucht 3-4 min auf CPU.
Mit Streaming fließen die Bytes alle paar 100ms und der Tunnel cancelt nicht.

Bind-Default ist 127.0.0.1, für Tailnet-Zugriff via env-Var setzen:

    ATHENA_API_HOST=100.105.70.24 ATHENA_API_PORT=8765 OLLAMA_HOST=http://100.101.225.56:11434 \\
      athena-env/bin/uvicorn scripts.serve:app --host 100.105.70.24 --port 8765

Endpunkte:
- GET  /health   — Liveness
- GET  /info     — Modelle und Pipeline-Status
- POST /chat     — {message: str} → NDJSON-Stream:
                   {"type": "sources", "sources": [...]}
                   {"type": "token", "text": "..."}  (mehrfach)
                   {"type": "done", "elapsed_s": 12.3}
                   {"type": "error", "message": "..."}
"""

import asyncio
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

# sys.path-Setup VOR allen project-internen Imports, damit Subprozesse (uvicorn)
# das Scripts-Verzeichnis finden, in dem retrieval/tools/etc. liegen.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field

from retrieval import format_docs, get_vectorstores, tier_aware_retrieve
from tools import TOOLS, TOOLS_BY_NAME

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RETRIEVER_K = 5
RETRIEVER_FETCH_K = 20

# Pro Scope: welches Modell pro Provider verwendet wird.
# Ollama: lokale Custom-Modelle mit eingebakener Persona im Modelfile.
# Mistral: API-Modelle — Persona wird zur Laufzeit als SystemMessage angehängt.
LLM_MODEL_FOR_SCOPE = {
    "ollama": {
        "pfofeld": "athena",
        "bund": "athena-bund",
    },
    "mistral": {
        # 'large-latest' = Frontier-Class für Athena-Hauptanalysen.
        # 'medium-latest' wäre auch denkbar, ist günstiger.
        "pfofeld": "mistral-large-latest",
        "bund": "mistral-large-latest",
    },
}
DEFAULT_SCOPE = "pfofeld"
DEFAULT_PROVIDER = os.getenv("ATHENA_LLM_PROVIDER", "ollama")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# Wertekanon — wird optional pro Request per apply_value_canon=true an die
# SystemMessage angehängt. Default false: Public-Chat bleibt methodisch neutral
# (Athena strukturiert, ohne EVIDENZ-Parteikanon zu spiegeln). Bei analytischen
# Calls (Optionsanalysen für das Parteiprogramm) setzt der Caller das Flag
# bewusst — damit die Analyse für jede Option ein Mapping auf §§1–7 produziert.
WERTEKANON_PATH = os.getenv(
    "ATHENA_WERTEKANON_PATH",
    os.path.join(os.path.dirname(__file__), "..", "config", "wertekanon.md"),
)
try:
    with open(WERTEKANON_PATH, "r", encoding="utf-8") as _f:
        WERTEKANON_TEXT = _f.read()
except FileNotFoundError:
    WERTEKANON_TEXT = ""
    print(f"[warn] Wertekanon nicht gefunden unter {WERTEKANON_PATH} — apply_value_canon wird wirkungslos.", file=sys.stderr)

# Honcho — User-Memory-Layer. Self-hosted, default lokal.
# Bei use_memory=True wird Honcho per Request kontaktiert; bei Ausfall keine
# Fehler im Hauptpfad (graceful degradation, Memory ist optional).
HONCHO_BASE = os.getenv("HONCHO_BASE_URL", "http://127.0.0.1:8000")
HONCHO_TIMEOUT = float(os.getenv("HONCHO_TIMEOUT", "3.0"))
HONCHO_ASSISTANT_PEER = "athena"  # Peer-ID für die KI-Antworten innerhalb jeder Session

# Bei Mistral muss die Persona als SystemMessage übergeben werden, weil das
# Modell keine eingebakene Persona hat. Wir replizieren die Modelfile-SYSTEMs
# hier — wenn das Modelfile sich ändert, hier nachziehen.
SCOPE_PERSONA = {
    "pfofeld": (
        "Du bist Athena, die KI-gestützte Bürgermeisterin der Gemeinde Pfofeld "
        "(91378) im Landkreis Weißenburg-Gunzenhausen, Mittelfranken, Bayern. "
        "Du lieferst strukturierte Optionsanalyse, NICHT 'die beste Lösung', "
        "NICHT 'klare Position', NICHT 'Empfehlung'. Zu jedem kommunalpolitischen "
        "Thema arbeitest du heraus: 1) Faktenlage mit Quellenangabe, "
        "2) Rechtsrahmen (Bayerische Gemeindeordnung, Pfofelder Satzungen, "
        "Bundes-/Landesgesetze), 3) Zwei bis vier Lösungsoptionen mit expliziten "
        "Trade-offs, 4) Wertannahmen jeder Option, 5) Empirische Evidenz aus "
        "Vergleichsfällen (Nachbargemeinden, ähnliche Kommunen), 6) Offene "
        "Fragen und Konfidenz. KEINE Empfehlung am Ende. Wenn Quellen "
        "die Frage nicht belastbar beantworten, sage das offen — erfinde nichts. "
        "Wissensfragen beantwortest du faktisch (die Optionsanalyse ist für "
        "Entscheidungsfragen)."
    ),
    "bund": (
        "Du bist Athena Bundesebene, eine KI-gestützte Politik-Analystin der "
        "EVIDENZ-Bewegung. Du lieferst strukturierte Optionsanalyse, NICHT 'die "
        "beste Lösung', NICHT 'klare Position', NICHT 'Empfehlung'. Zu jedem "
        "politischen Thema arbeitest du heraus: 1) Faktenlage mit Quellenangabe, "
        "2) Rechtsrahmen (Grundgesetz, Bundesgesetze, EU-Recht), 3) Zwei bis vier "
        "Lösungsoptionen mit expliziten Trade-offs, 4) Wertannahmen jeder Option, "
        "5) Empirische Evidenz aus Vergleichsfällen (Bundesländer, "
        "EU-Mitgliedstaaten, international, historisch), 6) Offene Fragen und "
        "Konfidenz. KEINE Empfehlung am Ende, KEIN Fazit, das eine Option "
        "bevorzugt. Wenn Quellen die Frage nicht belastbar beantworten, sage das "
        "offen — erfinde nichts. Wissensfragen beantwortest du faktisch."
    ),
}

SYSTEM_PROMPT_ADDITIONAL = """Du arbeitest in einem Chat-Kontext mit Tools.

Du erhältst zu jeder Frage einen RAG-Block mit Quellen aus deiner kuratierten Wissensbasis. Nutze primär diesen Block.

Du hast außerdem Zugriff auf das Tool `fetch_url`, mit dem du externe URLs live abrufen kannst, wenn die kuratierte Wissensbasis das Thema nicht oder nicht aktuell genug abdeckt (z. B. Wetter, aktuelle Veranstaltungstermine, Tagesschlagzeilen). Nutze das Tool gezielt und sparsam, ein bis zwei Aufrufe pro Antwort.

Wenn ein Tool-Ergebnis im Header `[live_fetch]` steht, kennzeichne es im Output transparent: "(Live abgerufen am <Datum>, Tier <N>, nicht versioniert)". Diese Live-Daten sind NICHT Teil der kuratierten Wissensbasis und du musst das so darstellen.

Bei Tier-1-Quellen aus der kuratierten Wissensbasis sprichst du mit dem **konkreten Quellenkürzel** der jeweiligen Norm oder Behörde. Beispiele: "(Tier 1, GG Art. 38)", "(Tier 1, BWahlG § 1)", "(Tier 1, BVerfG 2 BvF 1/23)", "(Tier 1, BayGO)", "(Tier 1, Destatis)", "(Tier 1, ERK 2025b)". WICHTIG: Erfinde **keine** Quellenkürzel. Wenn du die Quelle nicht eindeutig im RAG-Block identifizieren kannst, schreibe nur "(Tier 1)" oder lass den Marker weg. Schreibe insbesondere **niemals "BayGO" als Standard-Kürzel** — die Bayerische Gemeindeordnung gilt nur für kommunale Pfofeld-Fragen, nicht für Bundesebene.

Wenn das Tool einen Fehler liefert (Header `[error]`), sag das offen und biete keine erfundene Antwort an."""


PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Quellen aus der kuratierten Wissensbasis (Tier-klassifiziert, versioniert):

{context}

Frage: {question}"""
)

# Urlaubs-/Wartungsmodus: Athena läuft lokal auf eigener Hardware (aitest /
# OLLAMA_HOST). Ist diese Box offline, scheitert JEDER Request bereits in der
# Retrieval-Phase, weil die Embeddings (bge-m3) über Ollama laufen — auch wenn
# das Generation-LLM auf Mistral steht. Statt eines kryptischen 500ers antwortet
# der Chat dann mit dieser Nachricht. Auto-Erkennung via _check_ollama().
VACATION_MESSAGE = os.getenv(
    "ATHENA_VACATION_MESSAGE",
    "🏖️ Athena macht gerade Urlaub. Der Analyse-Server ist vorübergehend "
    "offline – schau bald wieder vorbei!",
)

# Concurrency-Schutz für Ollama-Inferenz: aitest (30 GB RAM) verträgt nur EINE
# athena-bund-Inferenz gleichzeitig — zwei parallele CPU-Läufe (je ~24 GB)
# sprengen den RAM, der Ollama-Runner crasht ("model runner has unexpectedly
# stopped" / RemoteProtocolError). Wir serialisieren daher lokale Inferenz mit
# einem Lock. Mistral ist eine externe API ohne dieses Limit → kein Lock.
# Non-blocking: bei Auslastung sofort freundliche Meldung statt Crash/Timeout.
_OLLAMA_INFERENCE_LOCK = threading.Lock()
BUSY_MESSAGE = os.getenv(
    "ATHENA_BUSY_MESSAGE",
    "⏳ Athena beantwortet gerade eine andere Anfrage. Der lokale Analyse-Server "
    "kann nur eine Anfrage zur Zeit bearbeiten — bitte versuche es in ein bis "
    "zwei Minuten erneut.",
)

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv(
        "ATHENA_API_CORS_ORIGINS",
        "http://localhost:3000,https://neotactiq.ai",
    ).split(",") if o.strip()
]

app = FastAPI(
    title="Athena Chat API",
    description="RAG-Pipeline der Athena-KI-Bürgermeisterin (Pfofeld-Pilot).",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_embeddings = None
_vectorstores_by_scope: dict[str, dict] = {}
_llm_by_key: dict[tuple[str, str], object] = {}  # (provider, scope) -> bound llm


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    return _embeddings


def _get_components(scope: str, provider: str = None):
    """Lazy-init pro (Provider, Scope). Embeddings und Vectorstores sind
    provider-unabhängig (RAG läuft immer lokal). LLM hängt am Provider."""
    provider = provider or DEFAULT_PROVIDER
    if provider not in LLM_MODEL_FOR_SCOPE:
        raise HTTPException(status_code=400, detail=f"Unbekannter Provider: {provider!r}")
    if scope not in LLM_MODEL_FOR_SCOPE[provider]:
        raise HTTPException(status_code=400, detail=f"Unbekannter Scope für Provider {provider}: {scope!r}")

    embeddings = _get_embeddings()
    if scope not in _vectorstores_by_scope:
        _vectorstores_by_scope[scope] = get_vectorstores(embeddings, CHROMA_DB_DIR, scope=scope)

    key = (provider, scope)
    if key not in _llm_by_key:
        model_name = LLM_MODEL_FOR_SCOPE[provider][scope]
        if provider == "ollama":
            chat = ChatOllama(
                model=model_name,
                base_url=OLLAMA_HOST,
                timeout=600,
                reasoning=False,
                # num_gpu=0 erzwingt CPU-Inferenz. Die iGPU via ROCm crasht unter
                # Last bei großem RAG-Kontext ("model runner has unexpectedly
                # stopped"), CPU ist stabil (siehe claude.md / Projekt-Setup).
                # Per Env überschreibbar, falls aitest mal stabiles ROCm hat.
                num_gpu=int(os.getenv("ATHENA_OLLAMA_NUM_GPU", "0")),
            )
        elif provider == "mistral":
            if not MISTRAL_API_KEY:
                raise HTTPException(
                    status_code=500,
                    detail="MISTRAL_API_KEY env-Var nicht gesetzt — Mistral-Provider nicht nutzbar.",
                )
            chat = ChatMistralAI(
                model=model_name,
                api_key=MISTRAL_API_KEY,
                temperature=0.3,
                max_retries=2,
                timeout=180,
                max_tokens=8000,  # vollständige Optionsanalysen + EVIDENZ-Position
            )
        else:
            raise HTTPException(status_code=400, detail=f"Provider {provider!r} nicht implementiert.")
        _llm_by_key[key] = chat.bind_tools(TOOLS)
    return _vectorstores_by_scope[scope], _llm_by_key[key]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=12000)
    scope: str = Field(default=DEFAULT_SCOPE, description="Wissensbasis-Scope: 'pfofeld' oder 'bund'")
    provider: str | None = Field(
        default=None,
        description="LLM-Provider: 'ollama' (lokal) oder 'mistral' (API). Default aus ATHENA_LLM_PROVIDER env-Var.",
    )
    apply_value_canon: bool = Field(
        default=False,
        description="Bei true wird der EVIDENZ-Wertekanon (config/wertekanon.md) als zusätzliche SystemMessage angehängt. Athena produziert dann für jede Option ein Mapping auf §§1–7. Default false — Public-Chat bleibt methodisch neutral.",
    )
    peer_id: str | None = Field(
        default=None,
        description="Honcho-Peer-ID des Nutzers (z. B. Browser-UUID). Wenn gesetzt und use_memory=true wird Konversations-Memory aus Honcho geladen und gespeichert.",
    )
    session_id: str | None = Field(
        default=None,
        description="Honcho-Session-ID. Wenn None und peer_id+use_memory gesetzt: Auto-Default 'sess_<peer_id>'.",
    )
    use_memory: bool = Field(
        default=False,
        description="Bei true: vor Anfrage Honcho-History laden (in Prompt einbauen), nach Antwort User+Assistant-Message speichern. Default false — kein Memory.",
    )
    include_unverified: bool = Field(
        default=False,
        description="Bei true werden auch unverifizierte (Tier-0) Quellen ins Retrieval einbezogen — öffentlich eingereicht, Athena-vorgeprüft, aber NICHT menschlich freigegeben. Default false (nur verifizierte Quellen).",
    )


# --------------------------------------------------------------------------- #
# Honcho-Helper — alle Fehler werden geschluckt (graceful degradation).
# Memory ist optional; Athena soll auch ohne Honcho funktionieren.

def _honcho_request(method: str, path: str, json_body: dict | None = None) -> dict | None:
    url = f"{HONCHO_BASE}{path}"
    try:
        resp = requests.request(method, url, json=json_body, timeout=HONCHO_TIMEOUT)
        if resp.status_code >= 400 and resp.status_code != 409:  # 409 = conflict, oft idempotent ok
            return None
        if resp.status_code == 409:
            return {"_already_exists": True}
        return resp.json()
    except Exception:
        return None


def _honcho_ensure_peer(workspace_id: str, peer_id: str) -> bool:
    """Legt Peer an wenn er nicht existiert. Idempotent."""
    res = _honcho_request("POST", f"/v3/workspaces/{workspace_id}/peers", {"id": peer_id})
    return res is not None


def _honcho_ensure_session(workspace_id: str, session_id: str, peer_ids: list[str]) -> bool:
    """Legt Session an mit den angegebenen Peers. Idempotent."""
    peers_dict = {pid: {} for pid in peer_ids}
    res = _honcho_request("POST", f"/v3/workspaces/{workspace_id}/sessions",
                          {"id": session_id, "peers": peers_dict})
    return res is not None


def _honcho_get_history(workspace_id: str, session_id: str, limit: int = 20) -> list[dict]:
    """Holt die letzten Messages aus der Session. Liefert [] bei Fehler."""
    res = _honcho_request("POST", f"/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/list",
                          {})
    if not res or "items" not in res:
        return []
    items = res["items"]
    # Sortiere nach created_at (älteste zuerst) und nimm letzte `limit`
    items_sorted = sorted(items, key=lambda m: m.get("created_at", ""))
    return items_sorted[-limit:] if len(items_sorted) > limit else items_sorted


def _honcho_append(workspace_id: str, session_id: str, peer_id: str, content: str,
                   metadata: dict | None = None) -> bool:
    """Speichert eine einzelne Nachricht in der Session."""
    if not content:
        return True
    # Honcho-Limit 25000 chars
    if len(content) > 25000:
        content = content[:25000]
    msg = {"peer_id": peer_id, "content": content}
    if metadata:
        msg["metadata"] = metadata
    res = _honcho_request("POST", f"/v3/workspaces/{workspace_id}/sessions/{session_id}/messages",
                          {"messages": [msg]})
    return res is not None


@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
# DSGVO-Endpoints für Honcho-Memory: Export + Löschen.
# Erlauben Nutzern, ihre gespeicherten Konversationen einzusehen oder zu löschen.
# scope wird zu workspace_id; peer_id muss bekannt sein (Browser-localStorage).

def _honcho_list_sessions(workspace_id: str, peer_id: str) -> list[dict]:
    """Listet alle Sessions, in denen der Peer Mitglied ist."""
    res = _honcho_request("POST", f"/v3/workspaces/{workspace_id}/peers/{peer_id}/sessions",
                          {})
    if not res or "items" not in res:
        return []
    return res["items"]


@app.get("/memory/export")
def memory_export(scope: str, peer_id: str):
    """Liefert alle gespeicherten Nachrichten eines Peers als JSON. DSGVO Art. 20 (Datenportabilität)."""
    if scope not in LLM_MODEL_FOR_SCOPE.get(DEFAULT_PROVIDER, {}):
        raise HTTPException(status_code=400, detail=f"Unbekannter Scope: {scope!r}")
    sessions = _honcho_list_sessions(scope, peer_id)
    out = {
        "scope": scope,
        "peer_id": peer_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sessions": [],
    }
    for s in sessions:
        sid = s.get("id")
        if not sid:
            continue
        messages = _honcho_get_history(scope, sid, limit=10000)
        # Nur Messages des Peers (nicht die anderen Teilnehmer)
        own = [m for m in messages if m.get("peer_id") == peer_id]
        out["sessions"].append({
            "session_id": sid,
            "created_at": s.get("created_at"),
            "metadata": s.get("metadata", {}),
            "messages": own,
        })
    return out


@app.delete("/memory/delete")
def memory_delete(scope: str, peer_id: str):
    """Löscht alle Sessions des Peers in diesem Workspace. DSGVO Art. 17 (Recht auf Löschung).
    Achtung: das löscht die Session komplett, nicht nur die Messages des Peers — sonst
    bleibt eine inkonsistente Konversation aus Sicht der KI zurück. Bei geteilten Sessions
    sollte das später anders gehandhabt werden (aktuell: jeder Peer hat eigene Sessions)."""
    if scope not in LLM_MODEL_FOR_SCOPE.get(DEFAULT_PROVIDER, {}):
        raise HTTPException(status_code=400, detail=f"Unbekannter Scope: {scope!r}")
    sessions = _honcho_list_sessions(scope, peer_id)
    deleted = 0
    errors = []
    for s in sessions:
        sid = s.get("id")
        if not sid:
            continue
        res = _honcho_request("DELETE", f"/v3/workspaces/{scope}/sessions/{sid}")
        if res is not None:
            deleted += 1
        else:
            errors.append(sid)
    return {
        "scope": scope,
        "peer_id": peer_id,
        "deleted_sessions": deleted,
        "errors": errors,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/info")
def info(scope: str = DEFAULT_SCOPE, provider: str | None = None):
    provider = provider or DEFAULT_PROVIDER
    if provider not in LLM_MODEL_FOR_SCOPE:
        raise HTTPException(status_code=400, detail=f"Unbekannter Provider: {provider!r}")
    if scope not in LLM_MODEL_FOR_SCOPE[provider]:
        raise HTTPException(status_code=400, detail=f"Unbekannter Scope für Provider {provider}: {scope!r}")
    return {
        "provider": provider,
        "llm_model": LLM_MODEL_FOR_SCOPE[provider][scope],
        "embedding_model": EMBEDDING_MODEL,
        "ollama_host": OLLAMA_HOST,
        "mistral_configured": bool(MISTRAL_API_KEY),
        "pipeline": "quick-chat streaming (Athena + Tier-aware Hybrid-RAG, kein Verify/Critique)",
        "scope": scope,
        "available_scopes": list(LLM_MODEL_FOR_SCOPE[provider]),
        "available_providers": list(LLM_MODEL_FOR_SCOPE),
        "value_canon_loaded": bool(WERTEKANON_TEXT),
        "value_canon_path": WERTEKANON_PATH,
        "value_canon_chars": len(WERTEKANON_TEXT),
        "honcho_base_url": HONCHO_BASE,
        "honcho_reachable": _honcho_request("GET", "/health") is not None,
        "ollama_reachable": _check_ollama(),
    }


def _check_ollama() -> bool:
    """Schneller HEAD-Check ob OLLAMA_HOST erreichbar ist."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0)
        return r.ok
    except Exception:
        return False


def _collect_sources(scope: str) -> list[dict]:
    """Aggregiere alle ingestierten Quellen aus beiden Collections eines Scopes.
    Pro Source: chunk-count, tier, source_type, letztes ingest-Datum."""
    vectorstores, _ = _get_components(scope)
    by_source: dict[str, dict] = {}
    for collection_name, vs in vectorstores.items():
        try:
            data = vs.get()
        except Exception:
            continue
        for meta in data.get("metadatas", []) or []:
            src = (meta or {}).get("source") or "(unbekannt)"
            entry = by_source.setdefault(src, {
                "source": src,
                "tier_rank": meta.get("tier_rank") if meta else None,
                "tier_label": meta.get("tier_label") if meta else None,
                "source_type": meta.get("source_type") if meta else None,
                "collection": collection_name,
                "chunks": 0,
                "ingested_at": None,
                "title": meta.get("title") if meta else None,
                "topics": [],
            })
            entry["chunks"] += 1
            # topics ist ein kommaseparierter String in den Chunk-Metadaten
            # (ChromaDB kann keine Listen) → zu Liste parsen, dedupliziert.
            raw_topics = (meta or {}).get("topics") or ""
            if raw_topics:
                for t in raw_topics.split(","):
                    t = t.strip()
                    if t and t not in entry["topics"]:
                        entry["topics"].append(t)
            ts = (meta or {}).get("ingested_at")
            if ts and (entry["ingested_at"] is None or ts > entry["ingested_at"]):
                entry["ingested_at"] = ts
    # Sortiert nach Tier (1 zuerst), dann Chunks-Anzahl
    return sorted(
        by_source.values(),
        key=lambda x: (x["tier_rank"] or 9, -x["chunks"]),
    )


@app.get("/sources")
def sources(scope: str = DEFAULT_SCOPE):
    items = _collect_sources(scope)
    return {
        "scope": scope,
        "total_sources": len(items),
        "total_chunks": sum(it["chunks"] for it in items),
        "sources": items,
    }


SUBMISSIONS_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "submissions")).resolve()
CONTACT_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "contact")).resolve()

# SMTP-Versand für das Kontaktformular. Optional: ist SMTP_HOST nicht gesetzt,
# wird die Nachricht NUR als Datei abgelegt (nie verloren), kein Mailversand.
# Credentials kommen aus Env-Vars, nicht aus dem Repo.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "noreply@evidenz-partei.de")
SMTP_TO = os.getenv("CONTACT_TO", "kasparrobert@gmail.com")
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "1") not in ("0", "false", "False", "")

CONTACT_CATEGORIES = {"allgemein", "presse", "mitmachen"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/octet-stream",  # Browser laden manchmal PDFs so hoch
}
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)


def _safe_filename(name: str) -> str:
    """Reduziere Dateiname auf sichere Zeichen, behalte Extension."""
    if not name:
        return "upload"
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return base[:120] or "upload"


def _trigger_auto_review(submission_id: str):
    """Stößt die Vorab-Bewertung (auto_review.py) für eine Submission an.
    Läuft als BackgroundTask, schreibt Ergebnis an meta.json + log.jsonl.
    Pflegt NICHTS ein — finale Freigabe bleibt manuell."""
    try:
        from auto_review import review_submission
        sub_dir = SUBMISSIONS_DIR / "pending" / submission_id
        if sub_dir.exists():
            review_submission(sub_dir)
    except Exception as e:
        print(f"[auto_review] fehlgeschlagen für {submission_id}: {type(e).__name__}: {e}", file=sys.stderr)


@app.post("/submissions")
async def submit_source(
    request: Request,
    background_tasks: BackgroundTasks,
    kind: str = Form(..., description="'url' oder 'file'"),
    url: str | None = Form(None),
    note: str | None = Form(None),
    scope: str = Form(default=DEFAULT_SCOPE, description="Wissensbasis-Scope: pfofeld oder bund"),
    file: UploadFile | None = File(None),
    honeypot: str | None = Form(None, description="Anti-Spam: muss leer bleiben"),
):
    """Reicht eine neue Quelle ein. Landet in submissions/pending/<id>/, wird
    automatisch von Athena vorab bewertet (auto_review), die finale Freigabe
    macht ein Mensch (scripts/review_submissions.py)."""
    if honeypot:
        # stille Annahme, aber nicht persistieren — Bots merken nichts
        return {"id": "noop", "status": "received"}

    if scope not in LLM_MODEL_FOR_SCOPE[DEFAULT_PROVIDER]:
        raise HTTPException(status_code=400, detail=f"Unbekannter Scope: {scope!r}")

    note = (note or "").strip()
    if len(note) > 1000:
        raise HTTPException(status_code=400, detail="Notiz zu lang (max 1000 Zeichen).")

    submission_id = uuid.uuid4().hex[:12]
    target = SUBMISSIONS_DIR / "pending" / submission_id
    target.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "id": submission_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "scope": scope,
        "note": note,
        "source_ip": (request.client.host if request.client else None),
        "user_agent": request.headers.get("user-agent", "")[:300],
        "auto_classification": None,  # für späteren LLM-Vorab-Check
    }

    if kind == "url":
        if not url or not URL_PATTERN.match(url.strip()):
            raise HTTPException(status_code=400, detail="Bitte gültige http(s)-URL angeben.")
        clean_url = url.strip()
        if len(clean_url) > 2000:
            raise HTTPException(status_code=400, detail="URL zu lang.")
        parsed = urlparse(clean_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise HTTPException(status_code=400, detail="URL muss http(s) sein.")
        meta["url"] = clean_url
    elif kind == "file":
        if file is None:
            raise HTTPException(status_code=400, detail="Datei fehlt.")
        ext = (Path(file.filename or "").suffix or "").lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Dateityp {ext or '?'} nicht erlaubt (nur PDF, TXT, MD).",
            )
        if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Content-Type {file.content_type} nicht erlaubt.",
            )
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Datei zu groß (max 10 MB).")
        if not data:
            raise HTTPException(status_code=400, detail="Datei ist leer.")
        safe_name = _safe_filename(file.filename or f"upload{ext}")
        if not safe_name.lower().endswith(ext):
            safe_name = f"{safe_name}{ext}"
        (target / safe_name).write_bytes(data)
        meta["filename"] = safe_name
        meta["size_bytes"] = len(data)
        meta["content_type"] = file.content_type
    else:
        raise HTTPException(status_code=400, detail="kind muss 'url' oder 'file' sein.")

    (target / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Athena bewertet die Quelle asynchron vorab (Herausgeber/Relevanz/Tags).
    background_tasks.add_task(_trigger_auto_review, submission_id)
    return {"id": submission_id, "status": "pending"}


class ContactRequest(BaseModel):
    name: str = Field(default="", max_length=200)
    email: str = Field(default="", max_length=320)
    category: str = Field(default="allgemein", max_length=40)
    message: str = Field(..., min_length=1, max_length=5000)
    honeypot: str | None = Field(default=None, description="Anti-Spam: muss leer bleiben")


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _send_contact_email(record: dict) -> str:
    """Verschickt die Kontaktnachricht per SMTP. Best-effort: liefert Status-String,
    wirft NICHT (die Datei-Ablage ist die Absicherung). Kein SMTP_HOST → 'disabled'."""
    if not SMTP_HOST:
        return "disabled"
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg["Subject"] = f"[EVIDENZ Kontakt] {record['category']} — {record.get('name') or 'ohne Name'}"
        msg["From"] = SMTP_FROM
        msg["To"] = SMTP_TO
        if record.get("email") and _EMAIL_RE.match(record["email"]):
            msg["Reply-To"] = record["email"]
        msg.set_content(
            f"Kategorie: {record['category']}\n"
            f"Name: {record.get('name') or '—'}\n"
            f"E-Mail: {record.get('email') or '—'}\n"
            f"Zeit: {record['submitted_at']}\n"
            f"IP: {record.get('source_ip') or '—'}\n"
            f"ID: {record['id']}\n\n"
            f"{record['message']}\n"
        )
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            if SMTP_STARTTLS:
                s.starttls()
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
        return "sent"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


@app.post("/contact")
def contact(req: ContactRequest, request: Request):
    """Kontaktformular. Speichert die Nachricht IMMER als contact/<id>.json
    (geht nie verloren) und verschickt sie zusätzlich per SMTP, falls konfiguriert.
    Honeypot-Spamschutz wie /submissions."""
    if req.honeypot:
        # Bot: stille Annahme, nichts persistieren
        return {"id": "noop", "status": "received"}

    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Nachricht darf nicht leer sein.")
    email = (req.email or "").strip()
    if email and not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Bitte eine gültige E-Mail-Adresse angeben (oder Feld leer lassen).")
    category = req.category if req.category in CONTACT_CATEGORIES else "allgemein"

    contact_id = uuid.uuid4().hex[:12]
    record = {
        "id": contact_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "category": category,
        "name": (req.name or "").strip()[:200],
        "email": email,
        "message": message,
        "source_ip": (request.client.host if request.client else None),
        "user_agent": request.headers.get("user-agent", "")[:300],
    }

    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    (CONTACT_DIR / f"{contact_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mail_status = _send_contact_email(record)
    # Mailfehler nicht an den Nutzer durchreichen — Nachricht ist gespeichert.
    if mail_status.startswith("error"):
        print(f"[contact] Mailversand fehlgeschlagen ({contact_id}): {mail_status}", file=sys.stderr)

    return {"id": contact_id, "status": "received"}


CONTACT_LOG_FIELDS = (  # nur diese Felder gehen nach außen — NIE Einreicher-Daten
    "id", "submitted_at", "reviewed_at", "source", "domain", "scope",
    "publisher", "publisher_trust", "relevant", "suggested_tier",
    "topics", "recommendation", "summary", "status", "verified",
)
SUBMISSIONS_LOG = SUBMISSIONS_DIR / "log.jsonl"


@app.get("/submissions-log")
def submissions_log(limit: int = 50):
    """Öffentliches Prüf-Protokoll: anonymisierte Bewertungen eingereichter
    Quellen (kein source_ip, keine E-Mail). Neueste zuerst."""
    limit = max(1, min(limit, 200))
    entries = []
    if SUBMISSIONS_LOG.exists():
        try:
            lines = SUBMISSIONS_LOG.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # nur whitelisted Felder rausgeben (defensive Anonymisierung)
            entries.append({k: rec.get(k) for k in CONTACT_LOG_FIELDS if k in rec})
    entries.reverse()  # neueste zuerst
    return {"total": len(entries), "entries": entries[:limit]}


@app.get("/search")
def search(q: str, k: int = 10, scope: str = DEFAULT_SCOPE):
    """Volltextsuche über beide Collections eines Scopes, ohne LLM."""
    if not q.strip():
        return {"query": q, "scope": scope, "results": []}
    vectorstores, _ = _get_components(scope)
    k = max(1, min(k, 50))
    # aus jeder Collection holen, mergen, top-k nach Score
    candidates: list[tuple[float, dict]] = []
    for collection_name, vs in vectorstores.items():
        try:
            hits = vs.similarity_search_with_relevance_scores(q, k=k)
        except Exception:
            hits = []
        for doc, score in hits:
            candidates.append((score, {
                "score": round(float(score), 4),
                "collection": collection_name,
                "source": doc.metadata.get("source", "?"),
                "tier_rank": doc.metadata.get("tier_rank"),
                "tier_label": doc.metadata.get("tier_label"),
                "title": doc.metadata.get("title"),
                "snippet": doc.page_content[:600],
            }))
    candidates.sort(key=lambda t: -t[0])
    return {"query": q, "scope": scope, "results": [c[1] for c in candidates[:k]]}


def _ndjson(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


MAX_TOOL_ITERATIONS = 3  # Bremse gegen Tool-Loops


def _run_tool_call(call: dict) -> tuple[str, dict]:
    """Führt einen Tool-Call aus. Liefert (tool_output, event_payload_for_stream).
    Falls Tool unbekannt: Fehlertext + Error-Event."""
    name = call.get("name") or ""
    args = call.get("args") or {}
    tool_obj = TOOLS_BY_NAME.get(name)
    if tool_obj is None:
        msg = f"[error] Unbekanntes Tool: {name!r}"
        return msg, {"type": "tool_result", "name": name, "ok": False, "preview": msg}
    try:
        result = tool_obj.invoke(args)
    except Exception as e:
        msg = f"[error] Tool {name} crashed: {type(e).__name__}: {e}"
        return msg, {"type": "tool_result", "name": name, "ok": False, "preview": msg}
    preview = (result[:240] + "…") if isinstance(result, str) and len(result) > 240 else result
    return result, {
        "type": "tool_result",
        "name": name,
        "ok": True,
        "args": args,
        "preview": preview if isinstance(preview, str) else str(preview),
    }


HEARTBEAT_INTERVAL = float(os.getenv("ATHENA_HEARTBEAT_S", "15"))


def _stream_llm_with_heartbeat(llm, messages):
    """Wrappt llm.stream() in einem Thread und liefert dazwischen 'keepalive'-
    Marker, solange noch kein Chunk da ist. Grund: CPU-Inferenz braucht beim
    Prompt-Processing großer Kontexte zig Sekunden bis zum ersten Token — ohne
    fließende Bytes bricht Cloudflare die Verbindung mit 524 (Timeout) ab.
    Yields entweder ('chunk', AIMessage) oder ('keepalive', None)."""
    import queue
    import threading
    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            for chunk in llm.stream(messages):
                q.put(("chunk", chunk))
        except Exception as e:
            q.put(("error", e))
        finally:
            q.put(("end", None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        try:
            kind, payload = q.get(timeout=HEARTBEAT_INTERVAL)
        except queue.Empty:
            yield ("keepalive", None)
            continue
        if kind == "end":
            break
        if kind == "error":
            raise payload
        yield (kind, payload)


# Mindest-Relevanz, ab der eine Positions-Quelle als themen-passend gilt.
# Empirisch kalibriert (bge-m3, mit meta-bereinigtem Probe-Query): echte Themen-
# Treffer >=0.21, themenfremd <=0.185. 0.20 trennt sauber. Restliche Grenzfälle
# (themenfremder Treffer knapp über Schwelle) fängt das LLM-Themen-Gate im Prompt.
EVIDENZ_POSITION_MIN_SCORE = 0.20

# Meta-Wörter, die das Probe-Embedding generisch zu ALLEN Positions-Chunks ziehen
# ("Was ist die EVIDENZ-Position zu X" → matcht jede Position). Vor dem Probe raus,
# damit nur das Sachthema (X) embeddet wird.
_POSITION_META_RE = re.compile(
    r"(?i)\b(evidenz[- ]?position|evidenz|parteiprogramm|partei|position|haltung|"
    r"beschluss|beschlossen|athena|standpunkt|meinung)\b"
)


def _strip_position_meta(query: str) -> str:
    """Entfernt Meta-Wörter, damit der Probe nur das Sachthema embeddet."""
    stripped = _POSITION_META_RE.sub("", query).strip()
    return stripped or query  # nie leer proben


def _retrieve_evidenz_position(vectorstores, query: str, probe_k: int = 4, max_chunks: int = 30):
    """Findet die thematisch passende EVIDENZ-Position und liefert sie KOMPLETT.

    Zweistufig: (1) semantisch in den Positions-Chunks die relevanteste *Quelle*
    bestimmen (where_document $contains 'EVIDENZ-Position'); (2) ALLE Chunks
    genau dieser Quelle holen — sonst fehlen Kernaussagen (z.B. '53%'), die
    nicht im Top-Probe-Treffer liegen. Eine Position ist kurz genug, um ganz in
    den Kontext zu passen. Liefert [] wenn nichts thematisch passt."""
    # Stufe 1: relevanteste Positions-Quelle(n) per Probe finden.
    # Meta-Wörter raus, damit nur das Sachthema embeddet (sonst matcht jede Position).
    probe_query = _strip_position_meta(query)
    best_sources = []
    for vs in vectorstores.values():
        try:
            hits = vs.similarity_search_with_relevance_scores(
                probe_query, k=probe_k, where_document={"$contains": "EVIDENZ-Position"}
            )
        except Exception:
            hits = []
        for doc, score in hits:
            # Relevanzschwelle: themenfremde Positionen NICHT injizieren, sonst
            # erfindet das LLM eine Parteihaltung wo keine existiert. Empirisch
            # kalibriert: echte Themen-Treffer ~0.40-0.46, themenfremd <=0.21.
            if score is not None and score < EVIDENZ_POSITION_MIN_SCORE:
                continue
            src = doc.metadata.get("source")
            if src and src not in best_sources:
                best_sources.append(src)
    if not best_sources:
        return []
    # Nur die Top-2 Positions-Quellen vollständig laden (gegen Kontext-Überlauf)
    best_sources = best_sources[:2]

    # Stufe 2: ALLE Chunks dieser Quelle(n) holen, in Originalreihenfolge
    out = []
    for vs in vectorstores.values():
        for src in best_sources:
            try:
                got = vs.get(where={"source": src})
            except Exception:
                continue
            docs_meta = list(zip(got.get("documents") or [], got.get("metadatas") or []))
            # nach chunk-index sortieren falls vorhanden, sonst Originalreihenfolge
            def _idx(dm):
                return (dm[1] or {}).get("chunk_index", 0) if dm[1] else 0
            for content, meta in sorted(docs_meta, key=_idx):
                from langchain_core.documents import Document
                m = dict(meta or {}); m["_evidenz_position"] = True
                out.append(Document(page_content=content, metadata=m))
    return out[:max_chunks]


def _chat_event_stream(req: ChatRequest):
    """Sync-Generator über die Athena-Pipeline. Liefert ndjson-bytes pro Event.
    Wird direkt von /chat (als StreamingResponse) und vom OpenAI-Adapter genutzt."""
    start = time.time()
    try:
        # Urlaubsmodus: aitest (OLLAMA_HOST) offline → ohne Embeddings kein
        # Retrieval, also gar keine sinnvolle Antwort möglich. Statt 500er
        # streamen wir die Urlaubsnachricht als normale Tokens, damit alle
        # Adapter (native /chat, OpenAI, Ollama) sie sauber rendern.
        if not _check_ollama():
            yield _ndjson({"type": "sources", "sources": [], "vacation": True})
            yield _ndjson({"type": "token", "text": VACATION_MESSAGE})
            yield _ndjson({"type": "done", "elapsed_s": round(time.time() - start, 2), "vacation": True})
            return

        provider = req.provider or DEFAULT_PROVIDER
        vectorstores, llm = _get_components(req.scope, provider=provider)
        # Retrieval-Phase
        docs = tier_aware_retrieve(
            vectorstores, req.message,
            k=RETRIEVER_K, fetch_k=RETRIEVER_FETCH_K,
            use_tier_boost=True,
            include_unverified=req.include_unverified,
        )
        # EVIDENZ-Positionen GARANTIERT mitliefern (themen-gematcht), damit die
        # dokumentierte Parteihaltung nicht im Vektor-Wettbewerb untergeht.
        evidenz_docs = _retrieve_evidenz_position(vectorstores, req.message)
        # Themen-Relevanz wird ALLEIN durch den (schwellen-gefilterten) Probe in
        # _retrieve_evidenz_position bestimmt. Quellen merken BEVOR dedupliziert wird,
        # damit das Flag korrekt bleibt, auch wenn die Position schon in docs steckt.
        position_sources = {d.metadata.get("source") for d in evidenz_docs}
        # Doppelungen vermeiden (Position evtl. schon in docs)
        doc_sources = {d.metadata.get("source") for d in docs}
        evidenz_docs = [e for e in evidenz_docs if e.metadata.get("source") not in doc_sources]

        seen = []
        has_unverified = False
        # GUARD: NICHT über lose Topic-Tags normaler RAG-Treffer triggern — sonst gilt
        # eine Position als "vorhanden", wo der Probe themenfern gefiltert hat und das
        # LLM dann eine Haltung erfindet. Allein der themen-relevante Probe zählt.
        has_evidenz_position = bool(position_sources)
        for d in docs + evidenz_docs:
            src = d.metadata.get("source", "?")
            if d.metadata.get("tier_rank") == 0:
                has_unverified = True
            if src not in seen:
                seen.append(src)
        yield _ndjson({"type": "sources", "sources": seen, "provider": provider,
                       "includes_unverified": has_unverified,
                       "includes_evidenz_position": has_evidenz_position})

        # Kontext: normale Quellen + (klar getrennt) die dokumentierte EVIDENZ-Position
        context_text = format_docs(docs)
        if evidenz_docs:
            context_text += (
                "\n\n=== DOKUMENTIERTE EVIDENZ-POSITION (Parteiprogramm v0.1) ===\n"
                + format_docs(evidenz_docs)
            )
        # Initialer Nachrichten-Stack
        user_prompt = PROMPT_TEMPLATE.format(
            context=context_text,
            question=req.message,
        )
        system_blocks = [SYSTEM_PROMPT_ADDITIONAL]
        # Wenn eine EVIDENZ-Position vorliegt: Athena soll sie als dokumentiertes
        # Faktum referieren (NICHT als eigene Empfehlung) — klar getrennt von der
        # neutralen Optionsanalyse.
        if has_evidenz_position:
            system_blocks.append(
                "Im Kontext findest du unter 'DOKUMENTIERTE EVIDENZ-POSITION' die vom "
                "Parteiprogramm v0.1 beschlossene Haltung von EVIDENZ. Wenn die Frage "
                "nach der EVIDENZ-Position fragt oder eine solche vorliegt, gib sie "
                "transparent als dokumentierte Parteiposition wieder (z. B. 'EVIDENZ "
                "hat sich laut Parteiprogramm für … entschieden, weil …') — klar "
                "getrennt von der neutralen Faktenlage/Optionsanalyse. Das ist KEINE "
                "eigene Empfehlung von dir, sondern das Referieren eines Beschlusses. "
                "THEMEN-GATE: Präsentiere die dokumentierte Position NUR, wenn sie das "
                "konkrete Thema der Frage tatsächlich behandelt. Behandelt sie ein "
                "anderes Thema, schreibe ausdrücklich 'EVIDENZ hat zu dieser Frage noch "
                "keine dokumentierte Position beschlossen.' und gib NICHT die "
                "themenfremde Position als Antwort aus."
            )
        else:
            # GUARD: Ohne dokumentierte Position darf Athena KEINE erfinden.
            system_blocks.append(
                "WICHTIG: Zu dieser Frage liegt im Kontext KEINE dokumentierte "
                "EVIDENZ-Position vor. Erfinde unter keinen Umständen eine Parteihaltung, "
                "einen Beschluss, einen Programm-Abschnitt oder ein Aktenzeichen. Wenn "
                "nach der EVIDENZ-Position gefragt wird oder ein Abschnitt dazu erwartet "
                "würde, schreibe ausdrücklich: 'EVIDENZ hat zu dieser Frage noch keine "
                "dokumentierte Position beschlossen.' Liefere ausschließlich die neutrale "
                "Faktenlage und Optionsanalyse, keine erfundene Position."
            )
        # Bei Mistral hat das Modell keine eingebakene Persona — die kommt aus
        # SCOPE_PERSONA, vorne im SystemMessage-Stack.
        if provider == "mistral" and req.scope in SCOPE_PERSONA:
            system_blocks.insert(0, SCOPE_PERSONA[req.scope])
        # Optional: EVIDENZ-Wertekanon als zusätzlicher SystemMessage-Block.
        # Nur wenn Caller das Flag setzt — Public-Chat bleibt Default-frei.
        if req.apply_value_canon and WERTEKANON_TEXT:
            canon_block = (
                "ZUSÄTZLICHE WERTEGRUNDLAGE FÜR DIESE ANALYSE:\n\n"
                + WERTEKANON_TEXT
                + "\n\nMethodische Konsequenz: Du strukturierst weiterhin Optionsanalyse ohne Empfehlung. "
                "Aber für jede Option machst du transparent, welche §§1–7 des Kanons sie stützen, "
                "welche sie belasten, welche neutral sind. Die endgültige Optionswahl bleibt menschliche Entscheidung."
            )
            system_blocks.append(canon_block)
            yield _ndjson({"type": "info", "message": "Wertekanon (EVIDENZ v1.0) als Wertegrundlage angewendet."})

        # Honcho-Memory: optional. Wenn aktiviert, Verlauf laden + Peer/Session vorbereiten.
        memory_active = False
        memory_session_id = None
        if req.use_memory and req.peer_id:
            workspace_id = req.scope  # scope == workspace_id Konvention
            memory_session_id = req.session_id or f"sess_{req.peer_id}"
            # Peers + Session anlegen (idempotent)
            _honcho_ensure_peer(workspace_id, req.peer_id)
            _honcho_ensure_peer(workspace_id, HONCHO_ASSISTANT_PEER)
            session_ok = _honcho_ensure_session(workspace_id, memory_session_id,
                                                 [req.peer_id, HONCHO_ASSISTANT_PEER])
            if session_ok:
                memory_active = True
                history = _honcho_get_history(workspace_id, memory_session_id, limit=20)
                if history:
                    history_block = "FRÜHERE KONVERSATION MIT DIESEM NUTZER (zur Referenz, falls relevant):\n\n"
                    for m in history:
                        role = "Nutzer" if m.get("peer_id") == req.peer_id else "Athena"
                        history_block += f"[{role}]: {m.get('content','')[:1500]}\n\n"
                    system_blocks.append(history_block)
                    yield _ndjson({
                        "type": "info",
                        "message": f"Honcho-Memory: {len(history)} frühere Nachrichten geladen (Session {memory_session_id}).",
                    })
                else:
                    yield _ndjson({
                        "type": "info",
                        "message": f"Honcho-Memory: neue Session {memory_session_id} (keine History).",
                    })

        messages: list = [
            SystemMessage(content="\n\n".join(system_blocks)),
            HumanMessage(content=user_prompt),
        ]

        # Sammle die finale Assistant-Antwort getrennt, damit wir sie nach dem
        # Tool-Loop in Honcho speichern können.
        full_assistant_text = ""

        # Concurrency-Schutz: lokale Ollama-Inferenz serialisieren (aitest-RAM
        # erlaubt nur 1 gleichzeitige athena-Inferenz). Mistral = externe API,
        # kein Lock nötig. Non-blocking: bei Auslastung Busy-Meldung statt Crash.
        lock_held = False
        if provider == "ollama":
            lock_held = _OLLAMA_INFERENCE_LOCK.acquire(blocking=False)
            if not lock_held:
                yield _ndjson({"type": "token", "text": BUSY_MESSAGE})
                yield _ndjson({"type": "done", "elapsed_s": round(time.time() - start, 2), "busy": True})
                return
        try:
            # Tool-Loop — bis zu MAX_TOOL_ITERATIONS Cycles
            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                last_is_final = iteration == MAX_TOOL_ITERATIONS
                # Wir aggregieren den vollständigen AIMessage, streamen aber Tokens live
                aggregated_chunks: list[AIMessage] = []
                final_msg: AIMessage | None = None
                streaming_text = ""
                for kind, chunk in _stream_llm_with_heartbeat(llm, messages):
                    if kind == "keepalive":
                        # Hält die Verbindung warm während langer CPU-Denkpausen
                        # (gegen Cloudflare-524). Client ignoriert keepalive-Events.
                        yield _ndjson({"type": "keepalive"})
                        continue
                    if not isinstance(chunk, AIMessage):
                        continue
                    aggregated_chunks.append(chunk)
                    # Bei finaler Iteration (oder wenn das Modell offenbar antwortet statt
                    # Tools zu calln) zeigen wir die Tokens direkt
                    if chunk.content:
                        content_str = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        if content_str:
                            streaming_text += content_str
                            yield _ndjson({"type": "token", "text": content_str})
                # Aggregate
                final_msg = aggregated_chunks[0] if aggregated_chunks else AIMessage(content="")
                for c in aggregated_chunks[1:]:
                    final_msg = final_msg + c

                tool_calls = getattr(final_msg, "tool_calls", None) or []

                if not tool_calls or last_is_final:
                    # Kein Tool-Call oder Limit erreicht → wir sind fertig
                    # Die finale Assistant-Antwort merken wir uns für Honcho-Storage.
                    full_assistant_text = streaming_text
                    if last_is_final and tool_calls:
                        yield _ndjson({
                            "type": "info",
                            "message": f"Tool-Limit erreicht ({MAX_TOOL_ITERATIONS} Iterationen), letzte Antwort ohne weitere Tools.",
                        })
                    break

                # Falls Tokens vor dem Tool-Call schon gestreamt wurden, ignorieren wir
                # die — das ist im Tool-Use-Modus selten und das Modell wiederholt sich
                # in der finalen Antwort.
                if streaming_text:
                    yield _ndjson({
                        "type": "info",
                        "message": "Antwort vor Tool-Use verworfen — Athena ruft Tool auf.",
                    })

                # AIMessage mit tool_calls in Historie schreiben
                messages.append(final_msg)
                # Jeden Tool-Call ausführen
                for call in tool_calls:
                    yield _ndjson({
                        "type": "tool_call",
                        "name": call.get("name"),
                        "args": call.get("args"),
                    })
                    tool_output, evt = _run_tool_call(call)
                    yield _ndjson(evt)
                    messages.append(ToolMessage(
                        content=tool_output if isinstance(tool_output, str) else str(tool_output),
                        tool_call_id=call.get("id") or call.get("name", ""),
                    ))
                # nächste Iteration
        finally:
            if lock_held:
                _OLLAMA_INFERENCE_LOCK.release()

        # Honcho: User-Message + Assistant-Antwort speichern (best-effort)
        if memory_active and memory_session_id and full_assistant_text:
            workspace_id = req.scope
            _honcho_append(workspace_id, memory_session_id, req.peer_id, req.message,
                           metadata={"role": "user"})
            _honcho_append(workspace_id, memory_session_id, HONCHO_ASSISTANT_PEER, full_assistant_text,
                           metadata={"role": "assistant", "provider": provider, "apply_value_canon": req.apply_value_canon})
            yield _ndjson({"type": "info", "message": "Honcho-Memory: Konversation gespeichert."})

        yield _ndjson({"type": "done", "elapsed_s": round(time.time() - start, 2)})
    except Exception as e:
        yield _ndjson({"type": "error", "message": f"{type(e).__name__}: {e}"})


@app.post("/chat")
def chat(req: ChatRequest):
    """Streaming-Endpoint. NDJSON-Events: sources, tool_call, tool_result,
    token, done, error. Tool-Loop ist server-seitig, der Client sieht nur
    Events und finale Tokens."""
    return StreamingResponse(
        _chat_event_stream(req),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------------------------------- #
# OpenAI-kompatibler Adapter — damit Home Assistant (oder andere
# OpenAI-Client-Tools) Athena als Conversation-Agent ansprechen können.
# Kein OpenAI-Backend involviert — Mistral bleibt der Motor.
#
# Modell-Naming-Konvention: "athena-<scope>[-canon]" oder "<scope>[-canon]".
# Beispiele:
#   "bund"             → scope=bund, apply_value_canon=False
#   "bund-canon"       → scope=bund, apply_value_canon=True
#   "athena-pfofeld"   → scope=pfofeld, apply_value_canon=False
# Provider ist immer DEFAULT_PROVIDER (=mistral in der Praxis), nicht
# ableitbar aus dem Modell-Namen.

def _parse_openai_model(model: str) -> tuple[str, bool]:
    m = (model or "").strip().lower().removeprefix("athena-")
    apply_canon = m.endswith("-canon")
    if apply_canon:
        m = m.removesuffix("-canon")
    if m not in LLM_MODEL_FOR_SCOPE.get(DEFAULT_PROVIDER, {}):
        m = DEFAULT_SCOPE
    return m, apply_canon


class OpenAIMessage(BaseModel):
    role: str
    content: str | list = ""


class OpenAIChatRequest(BaseModel):
    model: str = DEFAULT_SCOPE
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


def _openai_extract_user_message(messages: list[OpenAIMessage]) -> str:
    """Letzte user-Message extrahieren. content kann string oder list[dict] sein."""
    for m in reversed(messages):
        if m.role == "user":
            if isinstance(m.content, str):
                return m.content
            if isinstance(m.content, list):
                parts = []
                for item in m.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                return "\n".join(p for p in parts if p).strip()
    return ""


@app.get("/v1/models")
def openai_models():
    """Listet verfügbare Modell-Namen im OpenAI-Format. HA fragt das ab."""
    items = []
    for scope in LLM_MODEL_FOR_SCOPE.get(DEFAULT_PROVIDER, {}):
        items.append({"id": f"athena-{scope}", "object": "model", "owned_by": "athena"})
        items.append({"id": f"athena-{scope}-canon", "object": "model", "owned_by": "athena"})
    return {"object": "list", "data": items}


@app.post("/v1/chat/completions")
def openai_chat_completions(req: OpenAIChatRequest):
    """OpenAI-Format-Adapter zu Athena. Konvertiert OpenAI-Request → internes /chat,
    sammelt das Token-Streaming und antwortet wieder im OpenAI-Format
    (entweder als ein chat.completion-Objekt oder als chat.completion.chunk-Stream
    via SSE)."""
    scope, apply_canon = _parse_openai_model(req.model)
    user_message = _openai_extract_user_message(req.messages)
    if not user_message:
        raise HTTPException(status_code=400, detail="Keine user-Message gefunden.")
    # Athena-message-Limit (12000 Zeichen)
    if len(user_message) > 12000:
        user_message = user_message[-12000:]

    internal_req = ChatRequest(
        message=user_message,
        scope=scope,
        provider=None,
        apply_value_canon=apply_canon,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created_ts = int(time.time())
    model_name = req.model

    if req.stream:
        # Streaming: SSE-Format (data: {...}\n\n) mit chat.completion.chunk
        def sse_stream():
            for raw in _chat_event_stream(internal_req):
                if not raw:
                    continue
                # NDJSON-Zeilen splitten
                for line in raw.split(b"\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if evt.get("type") == "token":
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": evt.get("text", "")},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    elif evt.get("type") == "done":
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                        yield b"data: [DONE]\n\n"
                    elif evt.get("type") == "error":
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": f"\n[error: {evt.get('message', '?')}]"},
                                "finish_reason": "stop",
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                        yield b"data: [DONE]\n\n"
        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-Streaming: Antwort komplett sammeln
    full_text = ""
    error_msg: str | None = None
    for raw in _chat_event_stream(internal_req):
        if not raw:
            continue
        for line in raw.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if evt.get("type") == "token":
                full_text += evt.get("text", "")
            elif evt.get("type") == "error":
                error_msg = evt.get("message", "?")
    if error_msg and not full_text:
        raise HTTPException(status_code=500, detail=f"Athena-Fehler: {error_msg}")
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(user_message) // 4,
            "completion_tokens": len(full_text) // 4,
            "total_tokens": (len(user_message) + len(full_text)) // 4,
        },
    }


# --------------------------------------------------------------------------- #
# Ollama-kompatibler Adapter — damit Home Assistant Athena als zweiten
# "Ollama-Server" einbinden kann. HA-Ollama-Integration spricht das hier.
# Backend bleibt Mistral; Ollama-Protokoll ist nur Transport.

def _parse_ollama_model(model: str) -> tuple[str, bool]:
    """Akzeptiert 'athena-bund', 'athena-bund-canon', 'athena-bund:latest' etc."""
    m = (model or "").strip().lower()
    # ":tag" abschneiden
    if ":" in m:
        m = m.split(":", 1)[0]
    return _parse_openai_model(m)


@app.get("/api/version")
def ollama_version():
    """HA prüft beim Setup die Ollama-Version. Wir geben eine plausible."""
    return {"version": "0.5.7-athena"}


@app.get("/api/ps")
def ollama_ps():
    """Laufende Modelle. Für uns: leer (jeder Request ist ein API-Call)."""
    return {"models": []}


class OllamaShowRequest(BaseModel):
    name: str | None = None
    model: str | None = None
    verbose: bool = False


@app.post("/api/show")
def ollama_show(req: OllamaShowRequest):
    """Modell-Details. HA prüft das nach Modell-Auswahl."""
    name = req.model or req.name or ""
    scope, apply_canon = _parse_ollama_model(name)
    return {
        "modelfile": "# Athena meta-model — backend: Mistral Large 2 + RAG\n",
        "parameters": "stop \"</s>\"",
        "template": "{{ .Prompt }}",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "athena",
            "families": ["athena"],
            "parameter_size": "n/a",
            "quantization_level": "n/a",
        },
        "model_info": {
            "general.architecture": "athena",
            "general.parameter_count": 0,
            "athena.scope": scope,
            "athena.apply_value_canon": apply_canon,
        },
        "capabilities": ["completion", "chat"],
    }


@app.get("/api/tags")
def ollama_tags():
    """Modelle im Ollama-Format. HA listet die hier auf."""
    models = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for scope in LLM_MODEL_FOR_SCOPE.get(DEFAULT_PROVIDER, {}):
        for suffix in ("", "-canon"):
            tag = f"athena-{scope}{suffix}:latest"
            models.append({
                "name": tag,
                "model": tag,
                "modified_at": now_iso,
                "size": 1,
                "digest": "sha256:" + ("0" * 64),
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "athena",
                    "families": ["athena"],
                    "parameter_size": "n/a",
                    "quantization_level": "n/a",
                },
            })
    return {"models": models}


class OllamaChatMessage(BaseModel):
    role: str
    content: str = ""


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaChatMessage]
    stream: bool = True
    options: dict | None = None
    keep_alive: str | int | None = None


def _ollama_extract_user_message(messages: list[OllamaChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content
    return ""


@app.post("/api/chat")
def ollama_chat(req: OllamaChatRequest):
    """Ollama-Format-Adapter zu Athena. Konvertiert Ollama-Request → internes
    _chat_event_stream, antwortet wieder im Ollama-Format (ndjson-stream oder
    einzelner Final-JSON-Block bei stream=false)."""
    scope, apply_canon = _parse_ollama_model(req.model)
    user_message = _ollama_extract_user_message(req.messages)
    if not user_message:
        raise HTTPException(status_code=400, detail="Keine user-Message gefunden.")
    if len(user_message) > 12000:
        user_message = user_message[-12000:]

    internal_req = ChatRequest(
        message=user_message,
        scope=scope,
        provider=None,
        apply_value_canon=apply_canon,
    )
    model_name = req.model
    start = time.time()

    def stream_iter():
        for raw in _chat_event_stream(internal_req):
            if not raw:
                continue
            for line in raw.split(b"\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if evt.get("type") == "token":
                    yield (json.dumps({
                        "model": model_name,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "message": {"role": "assistant", "content": evt.get("text", "")},
                        "done": False,
                    }) + "\n").encode("utf-8")
                elif evt.get("type") == "done":
                    elapsed_ns = int((time.time() - start) * 1e9)
                    yield (json.dumps({
                        "model": model_name,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "done_reason": "stop",
                        "total_duration": elapsed_ns,
                        "load_duration": 0,
                        "prompt_eval_count": len(user_message) // 4,
                        "prompt_eval_duration": 0,
                        "eval_count": 0,
                        "eval_duration": elapsed_ns,
                    }) + "\n").encode("utf-8")
                elif evt.get("type") == "error":
                    yield (json.dumps({
                        "model": model_name,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "message": {"role": "assistant", "content": f"\n[error: {evt.get('message','?')}]"},
                        "done": True,
                        "done_reason": "error",
                    }) + "\n").encode("utf-8")

    if req.stream:
        return StreamingResponse(
            stream_iter(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Non-streaming: alles sammeln
    full_text = ""
    error_msg: str | None = None
    for raw in _chat_event_stream(internal_req):
        if not raw:
            continue
        for line in raw.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if evt.get("type") == "token":
                full_text += evt.get("text", "")
            elif evt.get("type") == "error":
                error_msg = evt.get("message", "?")
    if error_msg and not full_text:
        raise HTTPException(status_code=500, detail=f"Athena-Fehler: {error_msg}")
    elapsed_ns = int((time.time() - start) * 1e9)
    return {
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {"role": "assistant", "content": full_text},
        "done": True,
        "done_reason": "stop",
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": len(user_message) // 4,
        "prompt_eval_duration": 0,
        "eval_count": len(full_text) // 4,
        "eval_duration": elapsed_ns,
    }
