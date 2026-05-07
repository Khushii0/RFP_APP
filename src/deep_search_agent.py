"""
Deep Search Agent — multi-step research agent using LangChain deepagents.

Exposes:
  run_deep_search_agent(user_query)     → str  (blocking, returns final answer)
  stream_deep_search_agent(user_query)  → Generator[tuple, None, None]

Pattern follows: https://docs.langchain.com/oss/python/deepagents/deep-research
"""

import os
from datetime import datetime
from typing import Generator

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Overwrite

from src.prompts import (
    DEEP_RESEARCHER_PROMPT,
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)
from src.tools import DEEP_TOOLS, tavily_search, get_rfp_context, get_product_context

load_dotenv()

MAX_CONCURRENT_RESEARCH_UNITS = 3
MAX_RESEARCHER_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent():
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-5.4-nano"),
        temperature=0,
        
    )

    current_date = datetime.now().strftime("%Y-%m-%d")

    orchestrator_prompt = (
        RESEARCH_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
            max_concurrent_research_units=MAX_CONCURRENT_RESEARCH_UNITS,
            max_researcher_iterations=MAX_RESEARCHER_ITERATIONS,
        )
    )

    # Sub-agent used for focused research tasks
    research_sub_agent = {
        "name": "research-agent",
        "description": (
            "Delegate a focused research task to this sub-agent. "
            "It can search the web (Tavily) and load internal RFP/product data. "
            "Give one focused research topic at a time."
        ),
        "system_prompt": DEEP_RESEARCHER_PROMPT.format(date=current_date),
        "tools": [tavily_search, get_rfp_context, get_product_context],
    }

    return create_deep_agent(
        model=model,
        tools=DEEP_TOOLS,
        system_prompt=orchestrator_prompt,
        subagents=[research_sub_agent],
    )


# ---------------------------------------------------------------------------
# Helper: normalise message list from langgraph Overwrite wrapper
# ---------------------------------------------------------------------------

def _unwrap_messages(update: dict) -> list:
    messages = update.get("messages")
    if messages is None:
        return []
    if isinstance(messages, Overwrite):
        messages = messages.value
    return messages if isinstance(messages, list) else [messages]


def _reflect(user_query: str, task_findings: list[str]) -> str:
    """
    Post-hoc reflection call — separate, small LLM call.
    Its only job: read the sub-agent findings and explain WHY the final
    answer is correct in 3-4 sentences.
    """
    if not task_findings:
        return ""

    findings_text = "\n\n---\n\n".join(
        f"Finding {i+1}:\n{f[:800]}" for i, f in enumerate(task_findings)
    )

    reflection_prompt = (
        f"User asked: {user_query}\n\n"
        f"Sub-agents returned these findings:\n\n{findings_text}\n\n"
        "In 3-4 sentences, explain:\n"
        "- What the key evidence from these findings is\n"
        "- How it directly answers the user's question\n"
        "- Why the final conclusions / recommendations follow from this evidence\n\n"
        "Be concise. Do NOT repeat the full answer. Just explain the reasoning."
    )

    # Use a fast, cheap model for reflection — no need for the full model
    reflection_model = ChatOpenAI(
        model=os.getenv("REFLECTION_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")),
        temperature=0,
    )
    response = reflection_model.invoke(reflection_prompt)
    return response.content if isinstance(response.content, str) else str(response.content)


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

def stream_deep_search_agent(user_query: str) -> Generator[tuple, None, None]:
    """
    Runs the deep search agent and yields structured event tuples.

    Event tuple formats:
      ("plan",        todos: list[str])
      ("subagent",    description: str)
      ("tool_call",   tool_name: str, args: dict)
      ("tool_result", tool_name: str, preview: str)
      ("web_search",  query: str)
      ("reasoning",   text: str)          ← post-hoc reflection, emitted AFTER final_answer
      ("final_answer",text: str)
    """
    agent = _build_agent()
    final_answer = ""
    task_findings: list[str] = []   # collect sub-agent task results for reflection

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_query)]},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if not update:
                continue

            for msg in _unwrap_messages(update):

                # ── AI message ─────────────────────────────────────────────
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc["name"]
                            args = tc.get("args", {})

                            if name == "write_todos":
                                todos = args.get("todos", [])
                                if isinstance(todos, str):
                                    todos = [todos]
                                yield ("plan", todos)

                            elif name == "task":
                                desc = (
                                    args.get("description")
                                    or args.get("prompt")
                                    or str(args)
                                )
                                yield ("subagent", desc)

                            elif name == "tavily_search":
                                query_str = args.get("query", str(args))
                                yield ("web_search", query_str)
                                yield ("tool_call", name, args)

                            else:
                                yield ("tool_call", name, args)

                    elif msg.content:
                        # Clean AIMessage with no tool calls = orchestrator's
                        # final synthesised report. Store as final answer only.
                        text = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        final_answer = text

                # ── Tool result ─────────────────────────────────────────────
                elif isinstance(msg, ToolMessage):
                    raw = str(msg.content) if msg.content else ""
                    preview = raw[:400] + " …" if len(raw) > 400 else raw
                    yield ("tool_result", msg.name, preview)

                    # Collect sub-agent task findings for post-hoc reflection
                    if msg.name == "task" and raw:
                        task_findings.append(raw[:1200])  # cap per-finding size

    # ── Emit final answer to chat ───────────────────────────────────────────
    if final_answer:
        yield ("final_answer", final_answer)

    # ── Post-hoc reflection — genuine reasoning from sub-agent evidence ─────
    # This is a separate, dedicated LLM call. Its ONLY job is to explain WHY
    # the answer is correct based on what sub-agents found. It never sees the
    # final report — it only sees the raw evidence, so it can't just copy it.
    if task_findings:
        reasoning = _reflect(user_query, task_findings)
        if reasoning:
            yield ("reasoning", reasoning)


# ---------------------------------------------------------------------------
# Simple blocking call
# ---------------------------------------------------------------------------

def run_deep_search_agent(user_query: str) -> str:
    """Run deep search agent and return the final answer string."""
    final = ""
    for event in stream_deep_search_agent(user_query):
        if event[0] == "final_answer":
            final = event[1]
    return final or "Deep research completed but no final report was generated."
