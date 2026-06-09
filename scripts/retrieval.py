"""Athena - geteilte Retrieval-Bausteine für RAG-Pipelines.

Hybrid-RAG mit zwei ChromaDB-Collections pro Scope:
  - 'static': Rechtsgrundlagen, Satzungen, Primärquellen (Tier 1)
  - 'fresh':  News, aktuelle Berichte, Sekundär-/Kommentarquellen (Tier 2/3)

Scopes:
  - 'pfofeld': Pilot-Wissensbasis. Collections heißen weiterhin 'static'/'fresh'
    (kein Prefix für Backward-Kompatibilität — bestehende Daten bleiben).
  - 'bund':    Bundesweite Wissensbasis. Collections 'bund_static'/'bund_fresh'.

Tier-aware Re-Ranking nutzt die `tier_rank`-Metadaten, die `ingest.py` aus der
zum Scope passenden Tier-YAML schreibt. Boost-Werte sind LLM-/Domain-Tuning,
nicht Quellen-Pflege — daher hier im Code, nicht in der YAML.
"""

import os
import re

import chromadb
from langchain_chroma import Chroma

_HERE = os.path.dirname(__file__)
_DB_PATH = os.path.join(_HERE, "..", "athena-db")
_MODE_MARKER = os.path.join(_HERE, "..", ".chroma_mode")


def _chroma_mode() -> str:
    """'http' (Server-Modus) oder 'embedded'. Reihenfolge: Env ATHENA_CHROMA_HTTP
    überschreibt, sonst Marker-Datei .chroma_mode, sonst 'embedded' (Fallback).
    Die Marker-Datei sorgt dafür, dass ALLE Zugreifer (uvicorn + CLI-Skripte)
    denselben Modus nutzen — sonst öffnet ein Script embedded und kollidiert mit
    dem Server (genau die Korruption, die wir vermeiden wollen)."""
    v = os.getenv("ATHENA_CHROMA_HTTP")
    if v is not None:
        return "http" if v == "1" else "embedded"
    try:
        with open(_MODE_MARKER) as f:
            return f.read().strip().lower() or "embedded"
    except FileNotFoundError:
        return "embedded"


def chroma_server_mode() -> bool:
    """True, wenn ChromaDB im Server-Modus läuft. Dann sind parallele Writes
    sicher (der Server serialisiert) → CLI-Skripte müssen uvicorn NICHT stoppen."""
    return _chroma_mode() == "http"


def get_chroma_client():
    """Gemeinsamer ChromaDB-Client für uvicorn UND CLI-Skripte.
    Server-Modus → HttpClient (ein Besitzer-Server serialisiert Zugriffe →
    parallele Reads/Writes ohne Korruption, KEIN uvicorn-Stopp). Sonst embedded."""
    if _chroma_mode() == "http":
        return chromadb.HttpClient(
            host=os.getenv("ATHENA_CHROMA_HOST", "127.0.0.1"),
            port=int(os.getenv("ATHENA_CHROMA_PORT", "8001")),
        )
    return chromadb.PersistentClient(path=os.getenv("ATHENA_CHROMA_PATH") or _DB_PATH)

# Primärquellen schlagen knapp ähnlich-relevante Sekundärquellen, werden aber
# von deutlich besser passenden Tier-2/3-Chunks überstimmt (Soft-Preference).
# Tier 0 = unverifiziert (öffentlich eingereicht, Athena-vorgeprüft, aber NICHT
# menschlich freigegeben). Per Default vom Retrieval ausgeschlossen; nur mit
# include_unverified=True einbezogen, dann mit niedrigstem Boost.
TIER_BOOSTS = {0: 0.4, 1: 1.0, 2: 0.75, 3: 0.5}

# Hybride Suche: rein semantische Ähnlichkeit (bge-m3) verfehlt benannte
# Fachbegriffe, deren Quelltext metaphorisch/thematisch weit weg liegt — z. B.
# matcht "Rasenmähermethode" eher Landwirtschaft (Mähen) als die haushalts-
# politische Definition. Darum zusätzlich eine lexikalische $contains-Suche nach
# den seltenen Begriffen der Frage. Treffer bekommen einen Ähnlichkeits-Boden,
# weil ein exakter Begriffstreffer ein starkes Relevanzsignal ist, das der
# Embedding-Score unterschätzt.
LEXICAL_SIM_FLOOR = 0.6      # Mindest-"Ähnlichkeit" für exakte Begriffstreffer
LEXICAL_MIN_TERM_LEN = 7     # nur lange/seltene Tokens lexikalisch nachschlagen
LEXICAL_MAX_TERMS = 3        # höchstens so viele Begriffe pro Frage
LEXICAL_HITS_PER_TERM = 5    # höchstens so viele Treffer je Begriff/Collection
# Häufige lange Wörter, die KEINE Fachbegriffe sind (würden sonst Lärm ziehen).
_LEXICAL_STOPWORDS = {
    "welche", "welcher", "welches", "warum", "weshalb", "wieso", "stehst",
    "stehen", "meinst", "meinen", "bedeutet", "eigentlich", "zwischen",
    "möchte", "sollte", "sollten", "würde", "würden", "können", "könnte",
    "deiner", "deinem", "deinen", "unsere", "unserer", "unseren",
}


# Häufige Fugen-Endungen deutscher Komposita. Schreibt die Frage ein Wort
# ("Rasenmähermethode"), die Quelle aber getrennt/gebindestrichelt
# ("Rasenmäher-Methode"), findet ein $contains das Compound nicht — darum
# zusätzlich den distinktiven Wortstamm (Präfix vor der Endung) durchsuchen.
_COMPOUND_TAILS = (
    "methoden", "methode", "verfahren", "prinzipien", "prinzip", "modelle",
    "modell", "systeme", "system", "politik", "strategie", "konzept",
    "regelung", "ansatz", "ansätze", "satz", "bremse", "modell",
)


def _expand_compound(tok: str):
    """Den Begriff selbst liefern, plus — falls er auf eine bekannte Fugen-
    Endung endet — den vorderen Wortstamm (z. B. Rasenmähermethode→Rasenmäher)."""
    yield tok
    low = tok.lower()
    for tail in _COMPOUND_TAILS:
        if low.endswith(tail) and len(tok) - len(tail) >= 6:
            yield tok[: len(tok) - len(tail)]
            break


def _salient_terms(query: str) -> list[str]:
    """Seltene/lange Begriffe aus der Frage für die lexikalische Suche, inkl.
    Komposita-Stämme. Stoppwörter raus, Reihenfolge erhalten, dedupliziert,
    gekappt."""
    out = []
    seen = set()
    for tok in re.findall(r"[A-Za-zÄÖÜäöüß]{%d,}" % LEXICAL_MIN_TERM_LEN, query):
        for cand in _expand_compound(tok):
            low = cand.lower()
            if len(cand) < 6 or low in _LEXICAL_STOPWORDS or low in seen:
                continue
            seen.add(low)
            out.append(cand)
            if len(out) >= LEXICAL_MAX_TERMS:
                return out
    return out

SCOPES = ("pfofeld", "bund")


def collection_names_for(scope: str) -> list[str]:
    """Liefert die zwei Collection-Namen (static, fresh) für einen Scope.
    Pfofeld bleibt ohne Prefix wegen Backward-Kompatibilität."""
    if scope == "pfofeld":
        return ["static", "fresh"]
    return [f"{scope}_static", f"{scope}_fresh"]


# Backward-Compat-Konstante (alte Aufrufer ohne Scope = pfofeld)
COLLECTION_NAMES = collection_names_for("pfofeld")


def collection_for_source_type(source_type: str, scope: str = "pfofeld") -> str:
    """Mapping von source_type-Metadata auf Collection-Namen für einen Scope."""
    names = collection_names_for(scope)
    return names[0] if source_type == "static" else names[1]


def get_vectorstores(embeddings, persist_dir: str, scope: str = "pfofeld") -> dict[str, Chroma]:
    """Beide Collections eines Scopes instanziieren. Rückgabe-Keys sind die
    *kanonischen* Namen 'static'/'fresh' (damit Aufrufer scope-unabhängig
    arbeiten können), die tatsächlichen Collection-Namen werden über die
    Scope-Map aufgelöst."""
    real_names = collection_names_for(scope)
    if _chroma_mode() == "http":
        client = get_chroma_client()
        return {
            canonical: Chroma(
                client=client,
                collection_name=real,
                embedding_function=embeddings,
            )
            for canonical, real in zip(("static", "fresh"), real_names)
        }
    return {
        canonical: Chroma(
            collection_name=real,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        for canonical, real in zip(("static", "fresh"), real_names)
    }


def format_docs(docs):
    """Chunks für den LLM-Prompt zu einem Block joinen."""
    return "\n\n".join(d.page_content for d in docs)


def tier_aware_retrieve(
    vectorstores: dict[str, Chroma],
    query: str,
    k: int,
    fetch_k: int,
    use_tier_boost: bool = True,
    include_unverified: bool = False,
):
    """Aus jeder Collection top-fetch_k Kandidaten holen, mit Tier-Boost re-ranken,
    top-k zurückgeben. Bei use_tier_boost=False wird nur fusioniert, nicht gewichtet
    (für A/B-Vergleich mit reiner Vektor-Ähnlichkeit über beide Collections).

    include_unverified: Tier 0 (öffentlich eingereicht, nicht menschlich
    freigegeben) wird per Default AUSGESCHLOSSEN. Nur mit True einbezogen —
    schützt die normalen Analysen vor potenziell manipulierten Quellen.

    Erhält pro Doc:
      - doc.metadata['_collection']     : Herkunfts-Collection
      - doc.metadata['_similarity']     : Original-Similarity in [0, 1]
      - doc.metadata['_combined_score'] : Re-Rank-Score (sim * boost)
    """
    candidates = []
    for name, vs in vectorstores.items():
        try:
            cands = vs.similarity_search_with_relevance_scores(query, k=fetch_k)
        except Exception:
            cands = []
        for doc, similarity in cands:
            doc.metadata["_collection"] = name
            candidates.append((doc, similarity))

    # HYBRID: lexikalische $contains-Suche nach seltenen Begriffen der Frage.
    # Holt benannte Fachbegriffe in den Pool, die die Vektorsuche verfehlt.
    for term in _salient_terms(query):
        for name, vs in vectorstores.items():
            try:
                hits = vs.similarity_search_with_relevance_scores(
                    query, k=LEXICAL_HITS_PER_TERM,
                    where_document={"$contains": term},
                )
            except Exception:
                hits = []
            for doc, similarity in hits:
                doc.metadata["_collection"] = name
                doc.metadata["_lexical_term"] = term
                # exakter Begriffstreffer = starkes Signal → Ähnlichkeits-Boden
                candidates.append((doc, max(similarity, LEXICAL_SIM_FLOOR)))

    # Dedup: derselbe Chunk kann aus Vektor- UND lexikalischer Suche kommen
    # (oder mehrfach lexikalisch). Höchste Ähnlichkeit je Chunk behalten.
    best: dict[tuple, tuple] = {}
    for doc, similarity in candidates:
        key = (doc.metadata.get("source"), doc.metadata.get("chunk_index"),
               doc.page_content[:120])
        cur = best.get(key)
        if cur is None or similarity > cur[1]:
            best[key] = (doc, similarity)
    candidates = list(best.values())

    scored = []
    for doc, similarity in candidates:
        rank = doc.metadata.get("tier_rank", 3)
        # Tier 0 = unverifiziert: nur einbeziehen, wenn ausdrücklich gewünscht.
        if rank == 0 and not include_unverified:
            continue
        if not use_tier_boost:
            boost = 1.0
        elif rank == 0:
            # Unverifiziert: nach Athenas vorgeschlagenem Rang gewichten (aus dem
            # 'vorläufig-tierN'-Tag), aber gedämpft (×0.7), weil nicht menschlich
            # geprüft. Ohne Vorschlag → niedrigster Boost.
            boost = TIER_BOOSTS[0]
            sug = _suggested_tier_from_topics(doc.metadata.get("topics", ""))
            if sug in TIER_BOOSTS:
                boost = TIER_BOOSTS[sug] * 0.7
        else:
            boost = TIER_BOOSTS.get(rank, TIER_BOOSTS[3])
        combined = similarity * boost
        doc.metadata["_similarity"] = round(similarity, 4)
        doc.metadata["_combined_score"] = round(combined, 4)
        scored.append((doc, combined))
    scored.sort(key=lambda x: -x[1])
    return [d for d, _ in scored[:k]]


def _suggested_tier_from_topics(topics: str) -> int | None:
    """Liest Athenas vorgeschlagenen Rang aus einem 'vorläufig-tierN'-Tag im
    kommaseparierten topics-String. None, wenn keiner gesetzt ist."""
    if not topics:
        return None
    m = re.search(r"vorläufig-tier([123])", topics)
    return int(m.group(1)) if m else None
