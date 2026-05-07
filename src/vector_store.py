import json
import os
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load env
# ---------------------------------------------------------------------------

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent

_RFP_PATH = _BASE_DIR / "data" / "rfp_roadmap_strategy_full.json"

# ---------------------------------------------------------------------------
# Global stores
# ---------------------------------------------------------------------------

faiss_index = None

rfp_store = []
metadata_store = []

# ---------------------------------------------------------------------------
# Helper: Generate embeddings
# ---------------------------------------------------------------------------

def get_embedding(text: str):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    embedding = response.data[0].embedding

    return np.array(embedding, dtype="float32")

# ---------------------------------------------------------------------------
# Detect broad strategic enterprise questions
# ---------------------------------------------------------------------------

def is_broad_query(query: str) -> bool:

    broad_keywords = [
        "all rfps",
        "across rfps",
        "across all rfps",
        "capability gaps",
        "common requirements",
        "industry trends",
        "recommend product",
        "recommendation",
        "market needs",
        "enterprise needs",
        "what are the gaps",
        "compare",
        "comparison",
        "patterns",
        "strategic",
        "overall",
        "global",
        "which newgen products",
        "best products",
        "what capabilities",
        "coverage",
    ]

    q = query.lower()

    return any(keyword in q for keyword in broad_keywords)

# ---------------------------------------------------------------------------
# Ingest RFPs
# ---------------------------------------------------------------------------

def ingest_rfp_chunks():

    global faiss_index

    with open(_RFP_PATH, "r", encoding="utf-8") as f:
        rfp_data = json.load(f)

    rfp_store.clear()
    metadata_store.clear()

    embeddings_list = []

    # -------------------------------------------------------------------
    # ONE semantic chunk PER RFP
    # -------------------------------------------------------------------

    for idx, rfp in enumerate(rfp_data):

        rfp_name = (
            rfp.get("rfp_name")
            or rfp.get("name")
            or rfp.get("title")
            or f"RFP_{idx+1}"
        )

        industry = rfp.get("industry", "unknown")

        sections = []

        for key, value in rfp.items():

            if isinstance(value, str) and len(value.strip()) > 20:

                sections.append(
                    f"""
SECTION: {key.upper()}

{value}
"""
                )

        combined_text = "\n\n".join(sections)

        # ---------------------------------------------------------------
        # FULL ENTERPRISE RFP REPRESENTATION
        # ---------------------------------------------------------------

        full_rfp_text = f"""
RFP NAME: {rfp_name}

INDUSTRY: {industry}

ENTERPRISE REQUIREMENTS:

{combined_text}

This RFP may contain:
- business requirements
- integrations
- workflows
- operational goals
- compliance expectations
- automation objectives
- AI/ML requirements
- customer experience goals
- analytics requirements
- enterprise architecture expectations
"""

        rfp_store.append(full_rfp_text)

        metadata_store.append({
            "rfp_name": rfp_name,
            "industry": industry,
            "text": combined_text
        })

        embedding = get_embedding(full_rfp_text)

        embeddings_list.append(embedding)

    # -------------------------------------------------------------------
    # Create embedding matrix
    # -------------------------------------------------------------------

    embeddings = np.array(embeddings_list).astype("float32")

    faiss.normalize_L2(embeddings)

    # -------------------------------------------------------------------
    # Create FAISS index
    # -------------------------------------------------------------------

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(embeddings)

    print(f"[FAISS] Indexed {len(rfp_store)} FULL RFP semantic embeddings")

# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_rfp_knowledge(query: str, top_k: int = 10):

    global faiss_index

    if faiss_index is None:
        return "[ERROR] FAISS index not initialized"

    # -------------------------------------------------------------------
    # Broad strategic queries
    # -------------------------------------------------------------------

    if is_broad_query(query):

        query_embedding = get_embedding(query)

        query_embedding = np.array([query_embedding]).astype("float32")

        faiss.normalize_L2(query_embedding)

        # ---------------------------------------------------------------
        # Retrieve ALL RFPs ranked semantically
        # ---------------------------------------------------------------

        retrieve_count = len(metadata_store)

        distances, indices = faiss_index.search(
            query_embedding,
            retrieve_count
        )

        retrieved_rfps = []

        for idx in indices[0]:

            if idx >= len(metadata_store):
                continue

            metadata = metadata_store[idx]

            retrieved_rfps.append(
                f"""
RFP: {metadata['rfp_name']}
Industry: {metadata['industry']}

{metadata['text']}
"""
            )

        return "\n\n============================\n\n".join(retrieved_rfps)

    # -------------------------------------------------------------------
    # Specific focused queries
    # -------------------------------------------------------------------

    enhanced_query = f"""
Find highly relevant enterprise RFP information related to:

{query}

Focus on:
- feature requirements
- workflow requirements
- business needs
- integrations
- compliance expectations
- operational goals
- automation requirements
- enterprise pain points
"""

    query_embedding = get_embedding(enhanced_query)

    query_embedding = np.array([query_embedding]).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    retrieved_rfps = []

    seen_rfps = set()

    for idx in indices[0]:

        if idx >= len(metadata_store):
            continue

        metadata = metadata_store[idx]

        rfp_name = metadata["rfp_name"]

        if rfp_name in seen_rfps:
            continue

        seen_rfps.add(rfp_name)

        retrieved_rfps.append(
            f"""
RFP: {metadata['rfp_name']}
Industry: {metadata['industry']}

{metadata['text']}
"""
        )

    return "\n\n============================\n\n".join(retrieved_rfps)