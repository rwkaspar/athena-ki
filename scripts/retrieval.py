"""Athena - geteilte Retrieval-Bausteine für RAG-Pipelines.

Hybrid-RAG mit zwei ChromaDB-Collections:
  - 'static': Rechtsgrundlagen, Satzungen, Primärquellen (Tier 1)
  - 'fresh':  News, aktuelle Berichte, Sekundär-/Kommentarquellen (Tier 2/3)

Tier-aware Re-Ranking nutzt die `tier_rank`-Metadaten, die `ingest.py` aus
`config/source_tiers.yaml` schreibt. Boost-Werte sind LLM-/Domain-Tuning,
nicht Quellen-Pflege — daher hier im Code, nicht in der YAML.
"""

from langchain_chroma import Chroma

# Primärquellen schlagen knapp ähnlich-relevante Sekundärquellen, werden aber
# von deutlich besser passenden Tier-2/3-Chunks überstimmt (Soft-Preference).
TIER_BOOSTS = {1: 1.0, 2: 0.75, 3: 0.5}

COLLECTION_NAMES = ["static", "fresh"]


def collection_for_source_type(source_type: str) -> str:
    """Mapping von source_type-Metadata auf Collection-Namen."""
    return "static" if source_type == "static" else "fresh"


def get_vectorstores(embeddings, persist_dir: str) -> dict[str, Chroma]:
    """Beide Collections instanziieren. Leere Collections werden bei der
    ersten Schreib-/Leseoperation lazy angelegt."""
    return {
        name: Chroma(
            collection_name=name,
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        for name in COLLECTION_NAMES
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
):
    """Aus jeder Collection top-fetch_k Kandidaten holen, mit Tier-Boost re-ranken,
    top-k zurückgeben. Bei use_tier_boost=False wird nur fusioniert, nicht gewichtet
    (für A/B-Vergleich mit reiner Vektor-Ähnlichkeit über beide Collections).

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
        boost = TIER_BOOSTS.get(rank, TIER_BOOSTS[3]) if use_tier_boost else 1.0
        combined = similarity * boost
        doc.metadata["_similarity"] = round(similarity, 4)
        doc.metadata["_combined_score"] = round(combined, 4)
        scored.append((doc, combined))
    scored.sort(key=lambda x: -x[1])
    return [d for d, _ in scored[:k]]
