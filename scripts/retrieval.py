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

from langchain_chroma import Chroma

# Primärquellen schlagen knapp ähnlich-relevante Sekundärquellen, werden aber
# von deutlich besser passenden Tier-2/3-Chunks überstimmt (Soft-Preference).
# Tier 0 = unverifiziert (öffentlich eingereicht, Athena-vorgeprüft, aber NICHT
# menschlich freigegeben). Per Default vom Retrieval ausgeschlossen; nur mit
# include_unverified=True einbezogen, dann mit niedrigstem Boost.
TIER_BOOSTS = {0: 0.4, 1: 1.0, 2: 0.75, 3: 0.5}

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

    scored = []
    for doc, similarity in candidates:
        rank = doc.metadata.get("tier_rank", 3)
        # Tier 0 = unverifiziert: nur einbeziehen, wenn ausdrücklich gewünscht.
        if rank == 0 and not include_unverified:
            continue
        boost = TIER_BOOSTS.get(rank, TIER_BOOSTS[3]) if use_tier_boost else 1.0
        combined = similarity * boost
        doc.metadata["_similarity"] = round(similarity, 4)
        doc.metadata["_combined_score"] = round(combined, 4)
        scored.append((doc, combined))
    scored.sort(key=lambda x: -x[1])
    return [d for d, _ in scored[:k]]
