"""
tools.py => Tool definitions for the RFP Intelligence Agent.

Tools:
  - get_rfp_context      : Reads rfp_roadmap_strategy.json (~16 KB) → returns full string
  - get_product_context  : Reads newgen_products.json (~24 KB) → returns full string
  - tavily_search        : Live web search via Tavily API
"""

import json
import os
from pathlib import Path
from typing import Annotated, Literal

import httpx
from dotenv import load_dotenv
from langchain.tools import InjectedToolArg, tool
from markdownify import markdownify
from tavily import TavilyClient

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent.parent
_RFP_PATH = _BASE_DIR / "data" / "rfp_roadmap_strategy.json"
_PRODUCTS_PATH = _BASE_DIR / "data" / "newgen_products.json"

# ---------------------------------------------------------------------------
# Tavily client (lazy init — only if key present)
# ---------------------------------------------------------------------------

_tavily_client: TavilyClient | None = None


def _get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set in .env")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


# ---------------------------------------------------------------------------
# Helper: fetch a webpage and convert to markdown
# ---------------------------------------------------------------------------

def _fetch_page(url: str, timeout: float = 10.0) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        return markdownify(resp.text)
    except Exception as exc:
        return f"[Error fetching {url}: {exc}]"


# ---------------------------------------------------------------------------
# Tool 1 — get_rfp_context
# ---------------------------------------------------------------------------
from src.vector_store import search_rfp_knowledge


@tool
def get_rfp_context(query: str):
    """
    Retrieves enterprise-wide RFP intelligence using semantic retrieval.

    IMPORTANT BEHAVIOR:
    - For broad strategic/comparative questions:
      retrieves ALL relevant RFPs across the knowledge base.

    - For focused/specific questions:
      retrieves only the most relevant RFPs.

    This tool should be used whenever the user asks about:
    - RFP requirements
    - capability gaps
    - workflow expectations
    - integrations
    - compliance
    - feature requirements
    - enterprise needs
    - product fitment
    - comparison across RFPs
    - strategic recommendations
    - common industry patterns
    """

    try:

        retrieved_context = search_rfp_knowledge(
            query=query,
            top_k=10
        )

        # ---------------------------------------------------------------
        # Add retrieval guidance to LLM
        # ---------------------------------------------------------------

        final_context = f"""
IMPORTANT INSTRUCTIONS:

- The retrieved RFP context may contain MULTIPLE enterprise RFPs.
- For broad strategic/comparative questions:
  you MUST analyze patterns ACROSS all retrieved RFPs.
- Do NOT claim context is missing unless absolutely necessary.
- Do NOT answer from only one RFP if multiple RFPs are present.
- If multiple industries are present, compare them intelligently.

RETRIEVED RFP CONTEXT:

{retrieved_context}
"""

        return final_context

    except Exception as exc:
        return f"[ERROR] Failed to retrieve RFP data: {exc}"
# ---------------------------------------------------------------------------
# Tool 2 — get_product_context
# ---------------------------------------------------------------------------

@tool
def get_product_context() -> str:
    """
    Load and return the complete Newgen Software product portfolio data.

    Returns the full content of newgen_products.json as a formatted string (~24 KB).
    This includes all 8 Newgen modules:
    - NewgenONE Platform
    - NewgenONE Content ORB
    - NewgenONE Digital Process Automation Platform
    - NewgenONE Document Management System
    - Intelligent Document Processing (IDP)
    - NewgenONE Marvin (AI layer)
    - NewgenONE Integration Ecosystem

    Each module includes: core_problems, key_capabilities, features, use_cases, industries, keywords.

    Use this tool whenever the user asks about Newgen's existing products, capabilities,
    what Newgen already covers, or when comparing RFP needs against Newgen offerings.
    """
    try:
        with open(_PRODUCTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except FileNotFoundError:
        return f"[ERROR] Products file not found at: {_PRODUCTS_PATH}"
    except Exception as exc:
        return f"[ERROR] Failed to load product data: {exc}"


# ---------------------------------------------------------------------------
# Tool 3 — tavily_search
# ---------------------------------------------------------------------------

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[
        Literal["general", "news", "finance"],
        InjectedToolArg,
    ] = "general",
) -> str:
    """
    Search the web for live information on a given query using Tavily.

    Fetches real-time results and returns the full webpage content as markdown.
    Use this tool when you need:
    - Current industry trends and market data
    - Competitor product information
    - Emerging technology landscape
    - Any information that cannot be found in the internal RFP or product data

    In Deep Search mode, this tool MUST be called to enrich internal analysis with
    live market intelligence.

    Args:
        query: The search query string to execute.
        max_results: Number of results to return (default: 3).
        topic: Search topic filter — 'general', 'news', or 'finance' (default: 'general').

    Returns:
        Formatted string with search results including page titles, URLs, and full content.
    """
    try:
        client = _get_tavily_client()
        results = client.search(query, max_results=max_results, topic=topic)

        parts = []
        for item in results.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "No title")
            content = _fetch_page(url)
            parts.append(f"## {title}\n**URL:** {url}\n\n{content}\n---")

        if not parts:
            return f"No results found for query: '{query}'"

        return (
            f"Found {len(parts)} result(s) for '{query}':\n\n"
            + "\n".join(parts)
        )
    except Exception as exc:
        return f"[ERROR] Tavily search failed: {exc}"


# ---------------------------------------------------------------------------
# Exported tool lists per mode
# ---------------------------------------------------------------------------

NORMAL_TOOLS = [get_rfp_context, get_product_context]
DEEP_TOOLS = [get_rfp_context, get_product_context, tavily_search]
