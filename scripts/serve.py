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
import base64
import hashlib
import hmac
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
from fastapi.responses import JSONResponse, StreamingResponse
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
# Evidenz-Gate (Anti-Halluzination): Relevanz-Schwelle auf der Similarity.
# Empirisch kalibriert — relevante Fragen liegen ~0.6, themenfremder Unsinn ~0.25.
# Unter der Schwelle trägt keine Quelle → Athena schweigt, statt zu raten.
CHAT_SIM_FLOOR = 0.45
ABSTAIN_MESSAGE = (
    "Dazu finde ich in meiner kuratierten Quellenbasis **keine ausreichend belegten "
    "Quellen** — und ich rate nicht. Ich antworte nur, wenn Primärquellen (Gesetze, "
    "amtliche Statistik, Gutachten) die Aussage tragen. Frag mich gern anders formuliert "
    "oder enger auf ein Thema bezogen — oder schau direkt in die [Quellenbasis](/quellen.html)."
)

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

SYSTEM_PROMPT_ADDITIONAL = """ANSPRACHE (zwingend): Du sprichst IMMER mit einer Bürgerin oder einem Bürger, die/der eine Frage an EVIDENZ hat — allgemeinverständlich, sachlich, in der Ich-Form. Sprich dein Gegenüber durchgehend mit "du" an (Duzen), nie mit "Sie"; entsprechend "dir", "dein". Keine förmliche Begrüßung wie "Guten Tag" oder "Sehr geehrte/r". Du sprichst NIEMALS mit den Entwicklern oder Betreibern des Systems und unterstellst deinem Gegenüber nie, dass es das System gebaut hat oder Zugriff auf interne Daten hat.

KEINE INTERNE MECHANIK NENNEN: Erwähne in deiner Antwort keine technischen Interna (RAG, RAG-Block, Kontext-Block, Embeddings, Vektoren). Sage NIE "in deinem Kontext", "im vorliegenden Kontext", "im RAG-Block" o. Ä. Formuliere stattdessen natürlich für eine Bürgerin/einen Bürger: "nach den vorliegenden Quellen", "laut Parteiprogramm" oder "dazu liegen mir keine Quellen vor".

Du bekommst zu jeder Frage Quellen aus der kuratierten Wissensbasis mitgeliefert — stütze dich primär darauf.

Du hast außerdem das Tool `fetch_url`, mit dem du externe URLs live abrufen kannst, wenn die Wissensbasis ein Thema nicht oder nicht aktuell genug abdeckt (z. B. Wetter, aktuelle Termine, Tagesschlagzeilen). Nutze es gezielt und sparsam, ein bis zwei Aufrufe pro Antwort. Steht im Ergebnis der Header `[live_fetch]`, kennzeichne es transparent: "(Live abgerufen am <Datum>, nicht Teil der kuratierten Wissensbasis)". Bei `[error]` sag das offen und erfinde nichts.

Quellen-Marker: Bei Primärquellen nennst du das konkrete Quellenkürzel der Norm oder Behörde, z. B. "(Tier 1, GG Art. 38)", "(Tier 1, BWahlG § 1)", "(Tier 1, BVerfG 2 BvF 1/23)", "(Tier 1, Destatis)". Erfinde KEINE Kürzel — wenn du die Quelle nicht eindeutig zuordnen kannst, schreibe nur "(Tier 1)" oder lass den Marker weg. "BayGO" gilt nur für kommunale Pfofeld-Fragen, nie auf Bundesebene.

Wenn die vorliegenden Quellen eine Frage nicht belastbar beantworten oder thematisch nicht passen, sag das offen ("dazu liegen mir keine passenden Quellen vor") — gib NICHT themenfremde Inhalte als Antwort aus, nur weil sie zufällig mitgeliefert wurden."""


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
# Warteschlange: Anfragen warten der Reihe nach auf den einen Inferenz-Slot, statt
# abgewiesen zu werden. Tiefe begrenzt → kein unbegrenztes Auflaufen (Crash-Schutz).
_OLLAMA_QUEUE_LOCK = threading.Lock()
_OLLAMA_QUEUE_DEPTH = 0  # aktuell wartende + aktive Ollama-Anfragen
MAX_OLLAMA_QUEUE = int(os.getenv("ATHENA_MAX_QUEUE", "6"))  # 1 aktiv + 5 wartend
BUSY_MESSAGE = os.getenv(
    "ATHENA_BUSY_MESSAGE",
    "⏳ Athena beantwortet gerade eine andere Anfrage. Der lokale Analyse-Server "
    "kann nur eine Anfrage zur Zeit bearbeiten — bitte versuche es in ein bis "
    "zwei Minuten erneut.",
)
QUEUE_FULL_MESSAGE = os.getenv(
    "ATHENA_QUEUE_FULL_MESSAGE",
    "⏳ Die Warteschlange ist gerade voll — zu viele gleichzeitige Anfragen. "
    "Bitte versuche es in ein bis zwei Minuten erneut.",
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


def _ollama_keep_alive():
    """keep_alive für Ollama. -1/Zahlen → int (Ollama: -1 = für immer geladen,
    sonst Sekunden); Dauer-Strings wie '30m' bleiben String. Wichtig: der STRING
    '-1' wird von Ollama als Dauer fehlinterpretiert ('missing unit') → 400."""
    v = os.getenv("ATHENA_OLLAMA_KEEP_ALIVE", "-1")
    try:
        return int(v)
    except ValueError:
        return v


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_HOST,
            # Embedding-Modell ebenfalls warm halten (wird bei jedem Retrieval
            # gebraucht; Kaltstart = stille Wartezeit vor dem ersten Token).
            keep_alive=_ollama_keep_alive(),
        )
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
                # Niedrige Temperatur: ohne explizites Setzen nutzt Ollama Default
                # 0.8 → das Modell paraphrasiert und lässt bei jedem Lauf andere
                # konkrete Zahlen der EVIDENZ-Position weg (z.B. das 53%-Rentenniveau
                # nur in ~1/4 der Läufe). Empirisch belegt: bei 0.1-0.3 bleiben alle
                # Zahlen zuverlässig erhalten. Faktentreue > Kreativität.
                temperature=float(os.getenv("ATHENA_OLLAMA_TEMPERATURE", "0.15")),
                # Kontextfenster: muss groß genug sein, dass der komplette Prompt
                # (System + 3 Guards + bis zu 7 RAG-Quellen + volle EVIDENZ-Position
                # + Gesprächshistorie) NICHT das ganze Fenster füllt — sonst bleibt
                # kein Platz zum Generieren und die Antwort bricht nach 1 Token ab
                # (vage Folgefragen holen viele Quellen → 8192 reichte nicht). 16384
                # lässt ausreichend Generierungs-Spielraum; iGPU/GTT trägt den KV-Cache.
                num_ctx=int(os.getenv("ATHENA_OLLAMA_NUM_CTX", "16384")),
                # num_gpu=0 erzwingt CPU-Inferenz. Die iGPU via ROCm crasht unter
                # Last bei großem RAG-Kontext ("model runner has unexpectedly
                # stopped"), CPU ist stabil (siehe claude.md / Projekt-Setup).
                # Per Env überschreibbar, falls aitest mal stabiles ROCm hat.
                num_gpu=int(os.getenv("ATHENA_OLLAMA_NUM_GPU", "0")),
                # Modell geladen HALTEN. Ollama-Default (OLLAMA_KEEP_ALIVE=30s auf
                # aitest) entlädt das Modell zwischen Anfragen → jeder Besucher nach
                # einer Pause zahlt Kaltstart + Prefill (~112s gemessen) → so lange
                # Stille kappt die Verbindung (Cloudflare-524/Mobil). keep_alive=-1
                # hält NUR das Live-Modell warm (nicht Test-Modelle). RAM auf aitest
                # reicht (23 GB von 64 GB). Per Env anpassbar.
                keep_alive=_ollama_keep_alive(),
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
    history: list[dict] | None = Field(
        default=None,
        description="Client-seitiger Gesprächsverlauf: Liste von {role: 'user'|'assistant', content: str} der vorigen Turns. Wird als echte Dialog-Nachrichten vor die aktuelle Frage gestellt, damit Folgefragen Kontext haben — ohne server-seitiges Memory (Honcho).",
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
    try:
        import upload_security as _us
        clamd_ok = _us.clamd_available()
    except Exception:  # noqa: BLE001 — Health darf nie crashen
        clamd_ok = False
    return {"status": "ok", "clamd": "up" if clamd_ok else "down"}


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
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=4.0)
        return r.ok
    except Exception:
        return False


def _translation_index(scope: str, _cache: dict = {}) -> dict:
    """Map deutsches-Original → Liste von Übersetzungen: [{lang, url, title}].
    Wird beim /chat-Endpoint genutzt, um Sprachfassungen direkt in den
    Chat-Quellenlinks anzubieten („auch verfügbar in: 🇸🇦 ar"). Einfacher
    In-Process-Cache (max. 30 s) — bei jeder Inferenz-Antwort den ganzen
    Index zu scannen wäre verschwenderisch."""
    import time, re
    now = time.time()
    cached = _cache.get(scope)
    if cached and now - cached["ts"] < 30:
        return cached["idx"]
    from urllib.parse import unquote
    vectorstores, _ = _get_components(scope)
    idx: dict[str, list] = {}
    for collection_name, vs in vectorstores.items():
        offset, page = 0, 10000
        while True:
            try:
                data = vs.get(include=["metadatas"], limit=page, offset=offset)
            except Exception:
                break
            metas = data.get("metadatas") or []
            if not metas:
                break
            for m in metas:
                topics = (m or {}).get("topics") or ""
                if "translation_of:" not in topics:
                    if len(metas) < page:
                        break
                    continue
                lang_m = re.search(r"lang:([a-z]{2,3})", topics)
                trans_m = re.search(r"translation_of:([^,]+)", topics)
                if not trans_m:
                    continue
                orig = unquote(trans_m.group(1))
                src = (m or {}).get("source") or ""
                if not src:
                    continue
                entry = {
                    "lang": (lang_m.group(1) if lang_m else "?"),
                    "url": src,
                    "title": m.get("title"),
                }
                bucket = idx.setdefault(orig, [])
                # Dedup per (url, lang)
                if not any(b["url"] == entry["url"] and b["lang"] == entry["lang"] for b in bucket):
                    bucket.append(entry)
            if len(metas) < page:
                break
            offset += len(metas)
    _cache[scope] = {"ts": now, "idx": idx}
    return idx


def _collect_sources(scope: str) -> list[dict]:
    """Aggregiere alle ingestierten Quellen aus beiden Collections eines Scopes.
    Pro Source: chunk-count, tier, source_type, letztes ingest-Datum."""
    vectorstores, _ = _get_components(scope)
    by_source: dict[str, dict] = {}
    for collection_name, vs in vectorstores.items():
        # Paginiert lesen: ein einzelnes vs.get() über die ganze Collection sprengt
        # ab ~32k Records die SQLite-Variablen-Grenze ("too many SQL variables").
        # Nur Metadaten nötig (keine Dokumenttexte) → spart zusätzlich Speicher.
        metadatas = []
        offset, page = 0, 10000
        while True:
            try:
                data = vs.get(include=["metadatas"], limit=page, offset=offset)
            except Exception as e:
                print(f"[sources] {collection_name} get(offset={offset}) Fehler: "
                      f"{type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
                break
            batch = data.get("metadatas") or []
            metadatas.extend(batch)
            if len(batch) < page:
                break
            offset += len(batch)
        for meta in metadatas:
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
    # Verzeichnis wird erst nach bestandener Validierung/Scan angelegt (s. u.),
    # damit abgelehnte Uploads keine leeren Hüllen hinterlassen.

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

        # SCHUTZSCHICHT 1 — Virenscan der rohen Bytes (clamd, INSTREAM).
        # Vor jedem Schreiben aufs Dateisystem. Funde werden nie persistiert.
        import upload_security as _us
        try:
            _us.scan_bytes(data)
        except _us.InfectedError as e:
            raise HTTPException(
                status_code=422,
                detail=f"Datei abgelehnt: Schadsoftware erkannt ({e.signature}).",
            )
        except _us.ScanError as e:
            print(f"[upload] Virenscan nicht möglich: {e}", file=sys.stderr)
            raise HTTPException(
                status_code=503,
                detail="Virenscan derzeit nicht verfügbar — Upload vorübergehend "
                       "nicht möglich. Bitte später erneut versuchen.",
            )

        safe_name = _safe_filename(file.filename or f"upload{ext}")
        if not safe_name.lower().endswith(ext):
            safe_name = f"{safe_name}{ext}"

        target.mkdir(parents=True, exist_ok=True)
        if ext == ".pdf":
            # SCHUTZSCHICHT 2+3 — PDF NICHT im Serverprozess parsen, sondern im
            # isolierten Sandbox-Container; nur den Text behalten, das binäre
            # Original verwerfen (es wird nie aufs Dateisystem geschrieben).
            try:
                text = _us.extract_pdf_text(data)
            except _us.SandboxError as e:
                raise HTTPException(
                    status_code=422,
                    detail=f"PDF konnte nicht sicher verarbeitet werden: {e}",
                )
            text_name = f"{safe_name[:-4]}.txt"
            (target / text_name).write_text(text, encoding="utf-8")
            meta["filename"] = text_name
            meta["original_filename"] = safe_name
            meta["original_format"] = "pdf"
            meta["original_discarded"] = True   # Binär-PDF bewusst nicht gespeichert
            meta["pdf_text_extracted"] = True
            meta["sandbox_parsed"] = True
            meta["size_bytes"] = len(text.encode("utf-8"))
            meta["original_size_bytes"] = len(data)
            meta["content_type"] = "text/plain"
        else:
            # TXT/MD: bereits Text, durch den Virenscan gelaufen → unverändert ablegen.
            (target / safe_name).write_bytes(data)
            meta["filename"] = safe_name
            meta["size_bytes"] = len(data)
            meta["content_type"] = file.content_type
        meta["virus_scanned"] = True
    else:
        raise HTTPException(status_code=400, detail="kind muss 'url' oder 'file' sein.")

    target.mkdir(parents=True, exist_ok=True)  # idempotent; legt URL-Submissions-Dir an
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
    "reject_reason", "tier", "updated_at",
)
SUBMISSIONS_LOG = SUBMISSIONS_DIR / "log.jsonl"


@app.get("/submissions-log")
def submissions_log(limit: int = 50):
    """Öffentliches Prüf-Protokoll: anonymisierte Bewertungen eingereichter
    Quellen (kein source_ip, keine E-Mail). Neueste zuerst."""
    limit = max(1, min(limit, 200))
    # Pro Quelle (id) ALLE Log-Zeilen mergen — die Status-Updates (_log_status)
    # hängen nur Teilfelder an; spätere Zeilen gewinnen. So entsteht je Quelle EIN
    # aktueller Eintrag statt vieler Zwischenstände.
    merged: dict[str, dict] = {}
    order: list[str] = []
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
            sid = rec.get("id")
            if not sid:
                continue
            if sid not in merged:
                merged[sid] = {}
                order.append(sid)
            merged[sid].update({k: v for k, v in rec.items() if v is not None})
    entries = []
    for sid in order:
        rec = merged[sid]
        st = str(rec.get("status", ""))
        # Technische Ingest-Fehler sind KEINE Review-Entscheidung → nicht im Protokoll.
        if st.startswith("ingest_failed"):
            continue
        entries.append({k: rec.get(k) for k in CONTACT_LOG_FIELDS if k in rec})
    entries.reverse()  # neueste zuerst
    counts = {"verified": 0, "rejected": 0, "pending": 0}
    for e in entries:
        st = str(e.get("status", ""))
        if st == "rejected":
            counts["rejected"] += 1
        elif e.get("verified") or st == "verified":
            counts["verified"] += 1
        else:
            counts["pending"] += 1
    return {"total": len(entries), "counts": counts, "entries": entries[:limit]}


@app.get("/source-removals")
def source_removals(limit: int = 200):
    """Öffentliches Entfernungs-Protokoll: welche ECHTEN Quellen mit welcher
    Begründung aus der Wissensbasis entfernt wurden. Neueste zuerst.
    (Leerer Müll/Orphans wird bewusst NICHT protokolliert.)"""
    from source_audit import read_removals
    limit = max(1, min(limit, 1000))
    rows = read_removals(limit=limit)
    return {"total": len(rows), "removals": rows}


# --------------------------------------------------------------------------- #
# VERIFY: geschützte mobile Quellen-Freigabe (Phase 1 — Admin/Kern-Reviewer).
# Login per Passwort (Env), signiertes httpOnly-Session-Cookie. Schreibt über
# den Chroma-Server-Modus in die Live-Wissensbasis (kein uvicorn-Stopp nötig).
# Rollen-ready: Cookie trägt eine Rolle; advisory-Reviewer (Phase 2) kommen
# später als eigene Rolle dazu.
# --------------------------------------------------------------------------- #
VERIFY_PASSWORD = os.getenv("ATHENA_VERIFY_PASSWORD", "")
VERIFY_SECRET = os.getenv("ATHENA_VERIFY_SECRET") or (
    hashlib.sha256(("evidenz-verify::" + VERIFY_PASSWORD).encode()).hexdigest()
    if VERIFY_PASSWORD else "")
VERIFY_COOKIE = "evidenz_verify"
VERIFY_TTL = 7 * 24 * 3600

# Einfaches In-Memory-Rate-Limit (pro IP + Aktion). Gegen Registrierungs-Spam
# und Login-Brute-Force. Reicht für eine Single-Instance; bei Mehrfach-Workern
# müsste es geteilt werden (dann Redis o. Ä.).
_RL_BUCKETS: dict = {}


def _rate_ok(ip: str, action: str, max_n: int, window_s: int) -> bool:
    now = time.time()
    key = (action, ip or "?")
    b = _RL_BUCKETS.setdefault(key, [])
    b[:] = [t for t in b if now - t < window_s]
    if len(b) >= max_n:
        return False
    b.append(now)
    return True


def _verify_make_cookie(role: str, email: str = "") -> str:
    exp = int(time.time()) + VERIFY_TTL
    msg = f"{role}|{email}|{exp}"
    sig = hmac.new(VERIFY_SECRET.encode(), msg.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{msg}|{sig}".encode()).decode()


def _verify_session(request: Request):
    if not VERIFY_SECRET:
        return None
    raw = request.cookies.get(VERIFY_COOKIE)
    if not raw:
        return None
    try:
        dec = base64.urlsafe_b64decode(raw.encode()).decode()
        role, email, exp, sig = dec.split("|", 3)
        good = hmac.new(VERIFY_SECRET.encode(), f"{role}|{email}|{exp}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        if int(exp) < time.time():
            return None
        return {"role": role, "email": email}
    except Exception:
        return None


# --- Reviewer-Accounts (Phase 2): JSON-Store + Passwort-Hash + Mailversand ---
ACCOUNTS_FILE = SUBMISSIONS_DIR / "accounts.json"
SITE_BASE = os.getenv("EVIDENZ_SITE_BASE", "https://evidenz-partei.de")


def _accounts_load() -> dict:
    if ACCOUNTS_FILE.exists():
        try:
            return json.loads(ACCOUNTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _accounts_save(d: dict):
    ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNTS_FILE.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_pw(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 120000).hex()


def _send_email(to: str, subject: str, body: str) -> str:
    if not SMTP_HOST:
        return "disabled"
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_FROM
        msg["To"] = to
        msg.set_content(body)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
            if SMTP_STARTTLS:
                s.starttls()
            if SMTP_USER:
                s.login(SMTP_USER, SMTP_PASSWORD)
            s.send_message(msg)
        return "sent"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def _verify_require(request: Request) -> dict:
    s = _verify_session(request)
    if not s:
        raise HTTPException(status_code=401, detail="nicht angemeldet")
    return s


@app.post("/verify/login")
async def verify_login(request: Request):
    ip = request.client.host if request.client else "?"
    if not _rate_ok(ip, "login", 12, 600):
        raise HTTPException(status_code=429, detail="Zu viele Login-Versuche. Bitte später erneut.")
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    pw = body.get("password") or ""
    if not email:
        # Admin-Login (nur Passwort)
        if not VERIFY_PASSWORD or not hmac.compare_digest(str(pw), VERIFY_PASSWORD):
            raise HTTPException(status_code=401, detail="Falsches Passwort.")
        role, cookie_email = "admin", ""
    else:
        u = _accounts_load().get(email)
        if not u or not hmac.compare_digest(_hash_pw(str(pw), u["pw_salt"]), u["pw_hash"]):
            raise HTTPException(status_code=401, detail="E-Mail oder Passwort falsch.")
        if not u.get("verified"):
            raise HTTPException(status_code=403, detail="E-Mail noch nicht bestätigt.")
        # Rolle aus dem Account (server-seitig gesetzt; Selbstregistrierung bleibt 'reviewer').
        role, cookie_email = (u.get("role") or "reviewer"), email
    resp = JSONResponse({"ok": True, "role": role, "email": cookie_email})
    resp.set_cookie(VERIFY_COOKIE, _verify_make_cookie(role, cookie_email), max_age=VERIFY_TTL,
                    httponly=True, secure=True, samesite="lax", path="/")
    return resp


@app.get("/verify/me")
def verify_me(request: Request):
    s = _verify_session(request)
    return {"authed": bool(s), "role": (s or {}).get("role"),
            "email": (s or {}).get("email"), "configured": bool(VERIFY_SECRET)}


@app.post("/verify/logout")
def verify_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(VERIFY_COOKIE, path="/")
    return resp


def _user_languages(email: str | None) -> list[str]:
    """Sprachkenntnisse des Reviewers (lowercase ISO-Codes). Leer = unbekannt."""
    if not email:
        return []
    u = _accounts_load().get(email) or {}
    return [str(l).strip().lower() for l in (u.get("languages") or []) if l]


@app.get("/verify/profile")
def verify_profile(request: Request):
    """Eigenes Reviewer-Profil zurückgeben (aktuell: Sprachkenntnisse)."""
    sess = _verify_require(request)
    return {"email": sess.get("email"), "role": sess["role"],
            "languages": _user_languages(sess.get("email"))}


@app.post("/verify/profile/languages")
async def verify_profile_languages(request: Request):
    """Reviewer hinterlegt Sprachkenntnisse (z. B. ['de','ar','en']) — werden
    in der Pending-Queue benutzt, um fremdsprachige Submissions an passende
    Reviewer:innen zu routen. Speichert in accounts.json."""
    sess = _verify_require(request)
    email = sess.get("email")
    if not email:
        raise HTTPException(status_code=403, detail="Kein eigener Account.")
    body = await request.json()
    langs = body.get("languages") or []
    if not isinstance(langs, list):
        raise HTTPException(status_code=400, detail="languages muss Liste sein.")
    # ISO-Codes-Validierung (2-3 Zeichen, alpha)
    clean = []
    for l in langs[:20]:
        c = str(l).strip().lower()
        if 2 <= len(c) <= 3 and c.isalpha() and c not in clean:
            clean.append(c)
    users = _accounts_load()
    if email not in users:
        raise HTTPException(status_code=404, detail="Account nicht gefunden.")
    users[email]["languages"] = clean
    users[email]["languages_updated_at"] = datetime.now(timezone.utc).isoformat()
    _accounts_save(users)
    return {"ok": True, "languages": clean}


@app.get("/verify/pending")
def verify_pending(request: Request):
    sess = _verify_require(request)
    from review_submissions import list_pending, load_meta, _source_value
    is_admin = sess["role"] == "admin"
    # Sprach-Profil des Users — fremdsprachige Submissions werden danach
    # priorisiert. „de" und „en" sind implizit immer drin (Standard-RAG-Sprachen).
    user_langs = set(_user_languages(sess.get("email"))) | {"de", "en"}
    out = []
    for d in list_pending():
        m = load_meta(d)
        ar = m.get("auto_review") or {}
        votes = m.get("advisory_votes") or []
        # Default: advisory-Stimmen NUR für Admin sichtbar (kein Groupthink/Brigading).
        adv = ([{"vote": v.get("vote"), "reason": v.get("reason"), "email": v.get("email"),
                 "at": v.get("at")} for v in votes] if is_admin else [])
        my = next((v for v in votes if v.get("email") == sess.get("email")), None) if sess.get("email") else None
        out.append({
            "id": m.get("id"), "kind": m.get("kind"), "source": _source_value(m),
            "note": m.get("note"), "submitted_at": m.get("submitted_at"),
            "ingest_status": ar.get("ingest_status"),
            # Sprache: fremdsprachige Quellen werden markiert, nicht gefiltert —
            # ein sprachkundiger Reviewer kann sie bewerten/verlinken.
            "lang": m.get("lang") or ar.get("lang"),
            "lang_name": ar.get("lang_name"),
            "needs_language_review": bool(m.get("needs_language_review")),
            "is_translation_of": m.get("is_translation_of") or ar.get("is_translation_of"),
            "athena": {
                "recommendation": ar.get("recommendation"), "summary": ar.get("summary"),
                "publisher": ar.get("publisher"), "publisher_trust": ar.get("publisher_trust"),
                "relevant": ar.get("relevant"), "suggested_tier": ar.get("suggested_tier"),
                "topics": ar.get("topics"), "country": ar.get("country"),
            },
            "advisory_votes": adv, "vote_count": len(votes),
            "my_vote": {"vote": my.get("vote"), "reason": my.get("reason")} if my else None,
        })

    # Sortierung nach Sprach-Match: fremdsprachige Submissions, die genau die
    # Reviewer-Sprachen treffen, zuerst. Submissions in nicht-passenden
    # Fremdsprachen wandern ans Ende — sie warten auf einen anderen Helfer.
    def _lang_priority(item):
        l = (item.get("lang") or "").lower()
        if not item.get("needs_language_review"):
            return 0  # normale de/en-Submissions: Standardpriorität
        if l in user_langs:
            return -1  # passt zu eigener Sprache → ganz nach oben
        return 1  # fremde Sprache, die ich NICHT spreche → ans Ende
    out.sort(key=_lang_priority)
    return {"total": len(out), "pending": out, "role": sess["role"],
            "my_languages": sorted(user_langs)}


def _resolve_translation_original(scope: str, url: str) -> str:
    """Mappt die vom Reviewer angegebene Original-URL auf die tatsächlich
    ingestierte Quellen-URL — sonst verwaist die Sprachfassung auf quellen.html
    (das Einklappen unters Original braucht exaktes URL-Match).
    Reihenfolge: exakt > Slash-tolerant > Prefix (z. B. …/partg/ → …/partg/BJNR…html).
    Fällt auf die Eingabe zurück, wenn nichts passt."""
    if not url:
        return url
    norm = url.rstrip("/").lower()
    try:
        sources = [s.get("source") or "" for s in _collect_sources(scope)]
    except Exception:
        return url
    for s in sources:                                   # exakt / Slash-tolerant
        if s.rstrip("/").lower() == norm:
            return s
    cands = [s for s in sources if s.rstrip("/").lower().startswith(norm)]
    return min(cands, key=len) if cands else url        # kürzeste = kanonischste


@app.post("/verify/decide")
async def verify_decide(request: Request):
    sess = _verify_require(request)
    if sess["role"] != "admin":
        raise HTTPException(status_code=403, detail="Nur Admin/Kern darf final entscheiden.")
    body = await request.json()
    sid = (body.get("id") or "").strip()
    decision = (body.get("decision") or "").strip()
    reason = (body.get("reason") or "").strip()[:2000]
    tier_in = body.get("tier")
    from review_submissions import (load_meta, _update_tier0_chunks, _log_status,
                                    move_to, ingest_submission, APPROVED_DIR, REJECTED_DIR,
                                    CLARIFICATION_DIR, _set_source_lang_tags)
    sub_dir = SUBMISSIONS_DIR / "pending" / sid
    if not sid or not sub_dir.exists():
        raise HTTPException(status_code=404, detail="Submission nicht gefunden.")
    meta = load_meta(sub_dir)
    ar = meta.get("auto_review") or {}
    if decision == "ja":
        tier = int(tier_in) if str(tier_in) in ("1", "2", "3") else int(ar.get("suggested_tier") or 3)
        label = (ar.get("publisher") or "User-Submission")[:120]
        if ar.get("ingest_status") == "ingested_tier0":
            n = _update_tier0_chunks(meta, tier, label)
            ingest_note = f"{n} Chunks Tier0→{tier}"
            if n == 0:
                ingest_note = f"re-ingest rc={ingest_submission(sub_dir, meta, tier, label)}"
        else:
            ingest_note = f"ingest rc={ingest_submission(sub_dir, meta, tier, label)}"
        # Sprach-Behandlung: der Reviewer wählt die Bahn. Fremdsprachige Quellen
        # bekommen IMMER ein lang:-Tag (sonst lecken sie ins Retrieval). Bei
        # "translation" wird zusätzlich translation_of gesetzt (→ Bürger-Querlink
        # auf der Quellen-Seite, nicht für Athena), bei "standalone" entfernt.
        lang_code = (meta.get("lang") or ar.get("lang") or "").strip().lower()
        lang_mode = (body.get("lang_mode") or "").strip().lower()
        translation_of = (body.get("translation_of") or "").strip()
        if (lang_code and lang_code not in ("de", "unknown")) or lang_mode:
            to = "-"
            if lang_mode == "translation" and translation_of:
                to = _resolve_translation_original(meta.get("scope") or "pfofeld", translation_of)
            translation_of = to if to != "-" else translation_of
            n_tag = _set_source_lang_tags(meta, lang_code, to)
            ingest_note += (f" · lang:{lang_code or '—'}"
                            + (f" translation_of={translation_of}" if to != "-" else " (eigenständig)")
                            + f" [{n_tag} chunks]")
        move_to(sub_dir, APPROVED_DIR, extra_meta={
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "approved_tier": tier, "approved_label": label,
            "decided_by": sess["role"], "decision_reason": reason})
        _log_status(sid, "verified", {"verified": True, "tier": tier,
                    "decided_via": "web", "decision_reason": reason})
        return {"ok": True, "decision": "ja", "tier": tier, "ingest": ingest_note}
    if decision == "nein":
        if ar.get("ingest_status") == "ingested_tier0":
            _update_tier0_chunks(meta, -1, "")
        move_to(sub_dir, REJECTED_DIR, extra_meta={
            "rejected_at": datetime.now(timezone.utc).isoformat(),
            "reject_reason": reason, "decided_by": sess["role"]})
        _log_status(sid, "rejected", {"reject_reason": reason, "decided_via": "web"})
        return {"ok": True, "decision": "nein"}
    if decision == "klaerung":
        # Submission aus pending/ rausziehen — sonst taucht sie weiter in der
        # Review-Queue auf. Wartet in clarification/ bis Rückmeldung kommt
        # oder Admin sie über /verify/clarification/return wieder zurückholt.
        move_to(sub_dir, CLARIFICATION_DIR, extra_meta={
            "clarified_at": datetime.now(timezone.utc).isoformat(),
            "clarification_note": reason, "decided_by": sess["role"]})
        _log_status(sid, "needs_clarification", {"clarification_note": reason, "decided_via": "web"})
        return {"ok": True, "decision": "klaerung"}
    raise HTTPException(status_code=400, detail="decision muss ja|nein|klaerung sein.")


@app.post("/verify/register")
async def verify_register(request: Request):
    """Reviewer-Registrierung (auch parteifremd). E-Mail-Bestätigung nötig.
    Reviewer dürfen NUR advisory abstimmen, nicht final entscheiden."""
    ip = request.client.host if request.client else "?"
    if not _rate_ok(ip, "register", 3, 3600):
        raise HTTPException(status_code=429, detail="Zu viele Registrierungen. Bitte später erneut.")
    body = await request.json()
    email = (body.get("email") or "").strip().lower()
    pw = body.get("password") or ""
    if not body.get("consent"):
        raise HTTPException(status_code=400, detail="Bitte der Datenspeicherung zustimmen.")
    if not _EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Ungültige E-Mail.")
    if len(pw) < 8:
        raise HTTPException(status_code=400, detail="Passwort min. 8 Zeichen.")
    users = _accounts_load()
    ex = users.get(email)
    if ex and ex.get("verified"):
        raise HTTPException(status_code=409, detail="E-Mail bereits registriert.")
    salt = uuid.uuid4().hex
    token = uuid.uuid4().hex + uuid.uuid4().hex
    now_iso = datetime.now(timezone.utc).isoformat()
    users[email] = {"email": email, "pw_salt": salt, "pw_hash": _hash_pw(str(pw), salt),
                    "verified": False, "verify_token": token, "role": "reviewer",
                    "created_at": now_iso, "consent_at": now_iso}
    _accounts_save(users)
    link = f"{SITE_BASE}/api/athena/verify/confirm?token={token}&email={email}"
    sent = _send_email(email, "EVIDENZ — E-Mail bestätigen",
        "Hallo,\n\nbestätige deine Registrierung als Quellen-Reviewer bei EVIDENZ über diesen Link:\n"
        f"{link}\n\nAls Reviewer kannst du eingereichte Quellen bewerten (Pro/Contra/Unklar mit "
        "Begründung) — die finale Entscheidung trifft das Kernteam.\n\nWenn du das nicht warst, "
        "ignoriere diese Mail.\n")
    return {"ok": True, "email_status": sent}


@app.get("/verify/confirm")
def verify_confirm(token: str = "", email: str = ""):
    from fastapi.responses import HTMLResponse
    def page(msg):
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<body style='font-family:system-ui;background:#0e1116;color:#e6edf3;text-align:center;padding:3rem'>"
            f"<h2>{msg}</h2><p><a style='color:#f59e0b' href='/verify.html'>→ Zum Login</a></p>")
    users = _accounts_load()
    key = (email or "").strip().lower()
    u = users.get(key)
    if not token or not u or u.get("verify_token") != token:
        return page("Link ungültig oder bereits verwendet.")
    u["verified"] = True
    u["verify_token"] = None
    _accounts_save(users)
    return page("✓ E-Mail bestätigt. Du kannst dich jetzt anmelden.")


@app.post("/verify/vote")
async def verify_vote(request: Request):
    """Advisory-Stimme eines Reviewers (oder Admins): pro|contra|unklar + Begründung.
    Nicht bindend; wird an der Submission gespeichert."""
    sess = _verify_session(request)
    if not sess or sess["role"] not in ("reviewer", "admin"):
        raise HTTPException(status_code=401, detail="nicht angemeldet")
    body = await request.json()
    sid = (body.get("id") or "").strip()
    vote = (body.get("vote") or "").strip()
    reason = (body.get("reason") or "").strip()[:2000]
    if vote not in ("pro", "contra", "unklar"):
        raise HTTPException(status_code=400, detail="vote muss pro|contra|unklar sein.")
    from review_submissions import load_meta
    sub_dir = SUBMISSIONS_DIR / "pending" / sid
    if not sid or not sub_dir.exists():
        raise HTTPException(status_code=404, detail="Submission nicht gefunden.")
    meta = load_meta(sub_dir)
    voter = sess.get("email") or "admin"
    votes = [v for v in (meta.get("advisory_votes") or []) if v.get("email") != voter]
    votes.append({"email": voter, "vote": vote, "reason": reason,
                  "at": datetime.now(timezone.utc).isoformat()})
    meta["advisory_votes"] = votes
    (sub_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "vote": vote}


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
            sim_floor=CHAT_SIM_FLOOR,
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

        # ── Evidenz-Gate: Abstinenz statt Halluzination ──────────────────
        # Trägt KEINE Quelle die Frage (nichts über der Schwelle, keine themen-
        # relevante EVIDENZ-Position), antwortet Athena NICHT — sie sagt ehrlich,
        # dass die Beleglage fehlt. So kann keine Antwort auf erfundenen Quellen stehen.
        if not docs and not evidenz_docs:
            yield _ndjson({"type": "sources", "sources": [], "source_meta": {},
                           "provider": provider, "abstained": True})
            yield _ndjson({"type": "token", "text": ABSTAIN_MESSAGE})
            yield _ndjson({"type": "done", "elapsed_s": round(time.time() - start, 2),
                           "abstained": True})
            return

        seen = []
        has_unverified = False
        # GUARD: NICHT über lose Topic-Tags normaler RAG-Treffer triggern — sonst gilt
        # eine Position als "vorhanden", wo der Probe themenfern gefiltert hat und das
        # LLM dann eine Haltung erfindet. Allein der themen-relevante Probe zählt.
        has_evidenz_position = bool(position_sources)
        # source_meta: pro Quelle lesbarer Titel (für die Anzeige statt roher URL).
        # title = kurzer Anzeigetitel, title_full = vollständiger Originaltitel (Hover).
        source_meta = {}
        trans_idx = _translation_index(req.scope)
        for d in docs + evidenz_docs:
            src = d.metadata.get("source", "?")
            if d.metadata.get("tier_rank") == 0:
                has_unverified = True
            if src not in seen:
                seen.append(src)
            if src not in source_meta:
                m = d.metadata or {}
                source_meta[src] = {
                    "title": m.get("title") or None,
                    "title_full": m.get("title_full") or m.get("title") or None,
                    "tier_label": m.get("tier_label"),
                    # Übersetzungen dieser Quelle (für Bürger-Querverlinkung im Chat).
                    # RAG nutzt nur de/en; arabische/türkische/russische Fassungen
                    # tauchen hier nur als Service-Link für die Lesenden auf.
                    "translations": trans_idx.get(src, []),
                }
        # "+N weitere relevante Quellen": über dem Floor lagen mehr unabhängige
        # Quellen, als der Deckel zeigt — Transparenz statt stillem Abschneiden.
        more_sources = docs[0].metadata.get("_more_sources", 0) if docs else 0
        yield _ndjson({"type": "sources", "sources": seen, "source_meta": source_meta,
                       "provider": provider,
                       "includes_unverified": has_unverified,
                       "includes_evidenz_position": has_evidenz_position,
                       "more_sources": more_sources})

        # Kontext: normale Quellen + (klar getrennt) die dokumentierte EVIDENZ-Position.
        # CAP: vage Fragen holen viele/lange Quellen → Prompt kann num_ctx sprengen
        # (leere Antworten) und macht den Prefill auf CPU langsam. Deshalb deckeln —
        # die dokumentierte Position hat Vorrang (darf NICHT für RAG-Rauschen wegfallen,
        # sonst fehlen Kernzahlen wie 53 %), die normalen Docs werden zuerst gekürzt.
        NORMAL_DOCS_CHAR_BUDGET = int(os.getenv("ATHENA_CONTEXT_NORMAL_CHARS", "10000"))
        POSITION_CHAR_BUDGET = int(os.getenv("ATHENA_CONTEXT_POSITION_CHARS", "14000"))
        normal_text = format_docs(docs)
        if len(normal_text) > NORMAL_DOCS_CHAR_BUDGET:
            normal_text = normal_text[:NORMAL_DOCS_CHAR_BUDGET] + "\n…[gekürzt]"
        context_text = normal_text
        if evidenz_docs:
            pos_text = format_docs(evidenz_docs)
            if len(pos_text) > POSITION_CHAR_BUDGET:
                pos_text = pos_text[:POSITION_CHAR_BUDGET] + "\n…[gekürzt]"
            context_text += (
                "\n\n=== DOKUMENTIERTE EVIDENZ-POSITION (Parteiprogramm v0.1) ===\n"
                + pos_text
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
                "ZAHLEN-TREUE (zwingend): Übernimm ALLE konkreten Zahlen, Prozentsätze, "
                "Beträge und Jahreszahlen der dokumentierten Position WORTGETREU aus dem "
                "Kontext (z. B. Rentenniveau 53 %, Demografie-Beitrag 2,5 %, Beitragssatz "
                "21 %→18 %, Renteneintritt 67). Fasse die Position NICHT zusammenfassend "
                "zusammen und lasse KEINE der genannten Zahlen weg — fehlende Zahlen sind "
                "ein Fehler. Erfinde umgekehrt keine Zahlen, die nicht im Kontext stehen. "
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
        # ACHSEN-GUARD (immer aktiv): Fragen nach eindimensionaler ideologischer
        # Selbstverortung (links-rechts, progressiv-konservativ, "welches Lager",
        # "wie SPD/Grüne/CDU/AfD") darf Athena NICHT beantworten, indem sie EVIDENZ
        # selbst auf der Achse verortet — das wäre eine Wertung und widerspricht der
        # Methode. EVIDENZ lehnt die eindimensionale Achse bewusst ab.
        system_blocks.append(
            "ACHSEN-FRAMING (zwingend): Wenn nach der Einordnung von EVIDENZ auf einer "
            "Links-Rechts-Achse, im politischen Spektrum, in einem 'Lager', oder im "
            "Vergleich zu anderen Parteien (SPD, Grüne, CDU, FDP, AfD, Linke …) gefragt "
            "wird, ordne EVIDENZ NICHT selbst auf dieser Achse ein und vergib KEINE "
            "Partei-Etiketten ('links', 'linksliberal', 'konservativ', 'wie SPD/Grüne'). "
            "Übersetze auch den Wertekanon (§§) NICHT in Links-Rechts-Vokabular — der "
            "Kanon ist bewusst orthogonal zur Achse. Weise das eindimensionale Framing "
            "stattdessen transparent zurück: EVIDENZ verortet sich nicht auf der "
            "Links-Rechts-Achse, sondern bewertet jedes Politikfeld evidenzbasiert anhand "
            "der Wertannahmen des Kanons (§1–§7). Referiere dann die dokumentierten "
            "Feld-Positionen NEUTRAL und ohne Partei-Label. Stelle klar: Ob jemand eine "
            "einzelne Position EXTERN als 'links' oder 'rechts' liest, ist eine wertende "
            "Fremdzuschreibung, nicht Teil der EVIDENZ-Position. "
            "MEHRTURN-STANDHAFTIGKEIT (zwingend): Diese Regel gilt UNVERÄNDERT, auch wenn "
            "der Nutzer nachhakt, umformuliert oder über mehrere Nachrichten insistiert "
            "('doch', 'aber ungefähr', 'an den Werten gemessen', 'pro Politikfeld', 'wenn "
            "du müsstest'). Lass dich NICHT durch Nachdruck dazu bringen, EVIDENZ doch noch "
            "einzuordnen oder Felder einzelnen Parteien zuzuordnen ('Ähnlichkeit zu X', "
            "'liegt im Spektrum von Y', 'progressiv/konservativ wie Z'). Auch die Frage "
            "'gemessen am EVIDENZ-Wertekanon' beantwortest du über das Stützen/Belasten der "
            "§§1–7 — NICHT über eine Übersetzung in Links-Rechts oder Parteivergleiche. "
            "Bleibe bei jeder Wiederholung freundlich, aber konsequent bei der Zurückweisung "
            "des Achsen-Framings."
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

        # Client-seitiger Gesprächsverlauf (ohne Honcho): die vorigen Turns als
        # echte Dialog-Nachrichten VOR die aktuelle (RAG-augmentierte) Frage stellen.
        # Begrenzt auf die letzten Turns + Längen-Cap, damit num_ctx nicht überläuft.
        HISTORY_MAX_MSGS = 8        # letzte 8 Nachrichten (~4 Turns)
        HISTORY_MAX_CHARS = 2000    # pro Nachricht
        history_msgs: list = []
        if req.history:
            for turn in req.history[-HISTORY_MAX_MSGS:]:
                if not isinstance(turn, dict):
                    continue
                role = (turn.get("role") or "").lower()
                content = (turn.get("content") or "").strip()[:HISTORY_MAX_CHARS]
                if not content:
                    continue
                if role in ("user", "human"):
                    history_msgs.append(HumanMessage(content=content))
                elif role in ("assistant", "ai", "athena"):
                    history_msgs.append(AIMessage(content=content))
            if history_msgs:
                yield _ndjson({"type": "info",
                               "message": f"Gesprächskontext: {len(history_msgs)} vorige Nachricht(en) berücksichtigt."})

        messages: list = [
            SystemMessage(content="\n\n".join(system_blocks)),
            *history_msgs,
            HumanMessage(content=user_prompt),
        ]

        # Sammle die finale Assistant-Antwort getrennt, damit wir sie nach dem
        # Tool-Loop in Honcho speichern können.
        full_assistant_text = ""

        # Concurrency-Schutz: lokale Ollama-Inferenz serialisieren (aitest erlaubt
        # nur 1 gleichzeitige athena-Inferenz). Mistral = externe API, kein Lock.
        # Warteschlange statt Abweisung: Anfragen warten der Reihe nach; Tiefe ist
        # begrenzt (MAX_OLLAMA_QUEUE) → kein unbegrenztes Auflaufen (Crash-Schutz).
        lock_held = False
        queued = False
        if provider == "ollama":
            global _OLLAMA_QUEUE_DEPTH
            with _OLLAMA_QUEUE_LOCK:
                if _OLLAMA_QUEUE_DEPTH >= MAX_OLLAMA_QUEUE:
                    yield _ndjson({"type": "token", "text": QUEUE_FULL_MESSAGE})
                    yield _ndjson({"type": "done", "elapsed_s": round(time.time() - start, 2), "queue_full": True})
                    return
                _OLLAMA_QUEUE_DEPTH += 1
                position = _OLLAMA_QUEUE_DEPTH  # 1 = sofort dran
            queued = True
            if position > 1:
                yield _ndjson({"type": "info",
                               "message": f"In der Warteschlange (Position {position - 1} vor dir) — bleib dran."})
            # Blockierend mit Timeout warten, dabei Keepalive senden (gegen CF-524)
            while not lock_held:
                lock_held = _OLLAMA_INFERENCE_LOCK.acquire(timeout=HEARTBEAT_INTERVAL)
                if not lock_held:
                    yield _ndjson({"type": "keepalive"})
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
            if queued:
                with _OLLAMA_QUEUE_LOCK:
                    _OLLAMA_QUEUE_DEPTH -= 1

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


# --------------------------------------------------------------------------- #
# Resumable Chat-Jobs: Die Antwort läuft serverseitig in einem Hintergrund-Thread
# zu Ende und wird gepuffert — unabhängig davon, ob der Client verbunden bleibt.
# Bricht die Verbindung ab (Handy-Display aus, Netz-Blip), holt der Client die
# fehlenden Events per /chat/resume?job_id=…&offset=… nach. In-Memory + TTL
# (überlebt Client-Disconnect, NICHT einen Server-Neustart — für Mobile-Reconnect
# innerhalb von Minuten ausreichend).
_CHAT_JOBS: dict[str, dict] = {}
_CHAT_JOBS_LOCK = threading.Lock()
CHAT_JOB_TTL = float(os.getenv("ATHENA_CHAT_JOB_TTL", "900"))  # 15 min
_STREAM_HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}


def _cleanup_chat_jobs():
    now = time.time()
    for k in [k for k, j in _CHAT_JOBS.items() if now - j["ts"] > CHAT_JOB_TTL]:
        _CHAT_JOBS.pop(k, None)


def _run_chat_job(job_id: str, req: ChatRequest):
    """Läuft im Hintergrund-Thread bis zum Ende, puffert alle (nicht-keepalive)
    Events in den Job — auch wenn der Client längst weg ist."""
    job = _CHAT_JOBS[job_id]
    try:
        for chunk in _chat_event_stream(req):
            line = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            if '"keepalive"' in line:
                continue  # transient, nicht puffern (offset bleibt stabil)
            job["events"].append(line)
            job["ts"] = time.time()
    except Exception as e:
        job["events"].append(
            _ndjson({"type": "error", "message": f"interner Fehler: {type(e).__name__}"}).decode("utf-8")
        )
    finally:
        job["status"] = "done"
        job["ts"] = time.time()


def _tail_chat_job(job_id: str, offset: int = 0):
    """Streamt gepufferte Events ab `offset`; wartet auf neue, bis der Job fertig
    ist. Keepalive gegen CF-524-Timeout."""
    job = _CHAT_JOBS.get(job_id)
    if job is None:
        yield _ndjson({"type": "error", "message": "Sitzung abgelaufen — bitte neu fragen.", "expired": True})
        yield _ndjson({"type": "done", "elapsed_s": 0, "expired": True})
        return
    i = max(0, offset)
    last = time.time()
    while True:
        events = job["events"]
        if i < len(events):
            yield events[i].encode("utf-8")
            i += 1
            last = time.time()
            continue
        if job["status"] != "running":
            break
        if time.time() - last >= HEARTBEAT_INTERVAL:
            yield _ndjson({"type": "keepalive"})
            last = time.time()
        time.sleep(0.1)


@app.post("/chat")
def chat(req: ChatRequest):
    """Streaming-Endpoint (NDJSON: job, sources, token, done, error …).
    Die Antwort wird serverseitig gepuffert (resumable): erstes Event ist
    {type:'job', job_id}. Bricht die Verbindung ab, holt der Client den Rest
    via /chat/resume nach."""
    job_id = uuid.uuid4().hex
    with _CHAT_JOBS_LOCK:
        _cleanup_chat_jobs()
        _CHAT_JOBS[job_id] = {"events": [], "status": "running", "ts": time.time()}
    threading.Thread(target=_run_chat_job, args=(job_id, req), daemon=True).start()

    def gen():
        yield _ndjson({"type": "job", "job_id": job_id})
        yield from _tail_chat_job(job_id, 0)

    return StreamingResponse(gen(), media_type="application/x-ndjson", headers=_STREAM_HEADERS)


@app.get("/chat/resume")
def chat_resume(job_id: str, offset: int = 0):
    """Liefert die gepufferten Events eines laufenden/fertigen Jobs ab `offset`
    nach — für Reconnect nach Verbindungsabbruch."""
    return StreamingResponse(
        _tail_chat_job(job_id, offset), media_type="application/x-ndjson", headers=_STREAM_HEADERS
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
