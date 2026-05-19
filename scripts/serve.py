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

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# sys.path-Setup VOR allen project-internen Imports, damit Subprozesse (uvicorn)
# das Scripts-Verzeichnis finden, in dem retrieval/tools/etc. liegen.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field

from retrieval import format_docs, get_vectorstores, tier_aware_retrieve
from tools import TOOLS, TOOLS_BY_NAME

CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "athena-db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
RETRIEVER_K = 5
RETRIEVER_FETCH_K = 20

# Pro Scope: welches Ollama-Modell verwendet wird
LLM_MODEL_FOR_SCOPE = {
    "pfofeld": "athena",
    "bund": "athena-bund",
}
DEFAULT_SCOPE = "pfofeld"

SYSTEM_PROMPT_ADDITIONAL = """Du arbeitest in einem Chat-Kontext mit Tools.

Du erhältst zu jeder Frage einen RAG-Block mit Quellen aus deiner kuratierten Wissensbasis. Nutze primär diesen Block.

Du hast außerdem Zugriff auf das Tool `fetch_url`, mit dem du externe URLs live abrufen kannst, wenn die kuratierte Wissensbasis das Thema nicht oder nicht aktuell genug abdeckt (z. B. Wetter, aktuelle Veranstaltungstermine, Tagesschlagzeilen). Nutze das Tool gezielt und sparsam, ein bis zwei Aufrufe pro Antwort.

Wenn ein Tool-Ergebnis im Header `[live_fetch]` steht, kennzeichne es im Output transparent: "(Live abgerufen am <Datum>, Tier <N>, nicht versioniert)". Diese Live-Daten sind NICHT Teil der kuratierten Wissensbasis und du musst das so darstellen. Bei Tier-1-Quellen aus der kuratierten Wissensbasis sprichst du dagegen einfach von "(Tier 1, BayGO)" oder vergleichbar — ohne den Live-Marker.

Wenn das Tool einen Fehler liefert (Header `[error]`), sag das offen und biete keine erfundene Antwort an."""


PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Quellen aus der kuratierten Wissensbasis (Tier-klassifiziert, versioniert):

{context}

Frage: {question}"""
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
_llm_by_scope: dict[str, object] = {}


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)
    return _embeddings


def _get_components(scope: str):
    """Lazy-init pro Scope. Vectorstores und LLM-Bindings werden gecached."""
    if scope not in LLM_MODEL_FOR_SCOPE:
        raise HTTPException(status_code=400, detail=f"Unbekannter Scope: {scope!r}")
    embeddings = _get_embeddings()
    if scope not in _vectorstores_by_scope:
        _vectorstores_by_scope[scope] = get_vectorstores(embeddings, CHROMA_DB_DIR, scope=scope)
    if scope not in _llm_by_scope:
        chat = ChatOllama(
            model=LLM_MODEL_FOR_SCOPE[scope],
            base_url=OLLAMA_HOST,
            timeout=600,
            reasoning=False,
        )
        _llm_by_scope[scope] = chat.bind_tools(TOOLS)
    return _vectorstores_by_scope[scope], _llm_by_scope[scope]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    scope: str = Field(default=DEFAULT_SCOPE, description="Wissensbasis-Scope: 'pfofeld' oder 'bund'")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info(scope: str = DEFAULT_SCOPE):
    if scope not in LLM_MODEL_FOR_SCOPE:
        raise HTTPException(status_code=400, detail=f"Unbekannter Scope: {scope!r}")
    return {
        "llm_model": LLM_MODEL_FOR_SCOPE[scope],
        "embedding_model": EMBEDDING_MODEL,
        "ollama_host": OLLAMA_HOST,
        "pipeline": "quick-chat streaming (Athena + Tier-aware Hybrid-RAG, kein Verify/Critique)",
        "scope": scope,
        "available_scopes": list(LLM_MODEL_FOR_SCOPE),
    }


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
            })
            entry["chunks"] += 1
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


@app.post("/submissions")
async def submit_source(
    request: Request,
    kind: str = Form(..., description="'url' oder 'file'"),
    url: str | None = Form(None),
    note: str | None = Form(None),
    scope: str = Form(default=DEFAULT_SCOPE, description="Wissensbasis-Scope: pfofeld oder bund"),
    file: UploadFile | None = File(None),
    honeypot: str | None = Form(None, description="Anti-Spam: muss leer bleiben"),
):
    """Reicht eine neue Quelle ein. Landet in submissions/pending/<id>/, wird
    durch scripts/review_submissions.py manuell freigegeben."""
    if honeypot:
        # stille Annahme, aber nicht persistieren — Bots merken nichts
        return {"id": "noop", "status": "received"}

    if scope not in LLM_MODEL_FOR_SCOPE:
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
    return {"id": submission_id, "status": "pending"}


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


@app.post("/chat")
def chat(req: ChatRequest):
    """Streaming-Endpoint. NDJSON-Events: sources, tool_call, tool_result,
    token, done, error. Tool-Loop ist server-seitig, der Client sieht nur
    Events und finale Tokens."""

    def event_stream():
        start = time.time()
        try:
            vectorstores, llm = _get_components(req.scope)
            # Retrieval-Phase
            docs = tier_aware_retrieve(
                vectorstores, req.message,
                k=RETRIEVER_K, fetch_k=RETRIEVER_FETCH_K,
                use_tier_boost=True,
            )
            seen = []
            for d in docs:
                src = d.metadata.get("source", "?")
                if src not in seen:
                    seen.append(src)
            yield _ndjson({"type": "sources", "sources": seen})

            # Initialer Nachrichten-Stack
            user_prompt = PROMPT_TEMPLATE.format(
                context=format_docs(docs),
                question=req.message,
            )
            messages: list = [
                SystemMessage(content=SYSTEM_PROMPT_ADDITIONAL),
                HumanMessage(content=user_prompt),
            ]

            # Tool-Loop — bis zu MAX_TOOL_ITERATIONS Cycles
            for iteration in range(MAX_TOOL_ITERATIONS + 1):
                last_is_final = iteration == MAX_TOOL_ITERATIONS
                # Wir aggregieren den vollständigen AIMessage, streamen aber Tokens live
                aggregated_chunks: list[AIMessage] = []
                final_msg: AIMessage | None = None
                streaming_text = ""
                for chunk in llm.stream(messages):
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

            yield _ndjson({"type": "done", "elapsed_s": round(time.time() - start, 2)})
        except Exception as e:
            yield _ndjson({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
