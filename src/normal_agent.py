"""
Normal Mode Agent — ReAct agent using LangGraph.

Exposes:
  run_normal_agent(user_query)          → str  (blocking, returns final answer)
  stream_normal_agent(user_query)       → Generator[tuple, None, None]
"""

import os
from typing import Generator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.prompts import NORMAL_SYSTEM_PROMPT
from src.tools import NORMAL_TOOLS

load_dotenv()


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent():
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-5.4-nano"),
        temperature=0,
        streaming=True,
        
    )
    return create_react_agent(
        model,
        NORMAL_TOOLS,
        prompt=NORMAL_SYSTEM_PROMPT,   # LangGraph 1.x: state_modifier → prompt
    )


# ---------------------------------------------------------------------------
# Streaming generator
# ---------------------------------------------------------------------------

def stream_normal_agent(user_query: str) -> Generator[tuple, None, None]:
    """
    Runs the normal ReAct agent and yields structured event tuples.

    Event tuple formats:
      ("tool_call",   tool_name: str, args: dict)
      ("tool_result", tool_name: str, preview: str)
      ("reasoning",   text: str)
      ("final_answer",text: str)
    """
    agent = _build_agent()
    final_answer = ""

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_query)]},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            messages = update.get("messages", [])

            for msg in messages:
                # ── AI message ─────────────────────────────────────────────
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield ("tool_call", tc["name"], tc.get("args", {}))
                    elif msg.content and node == "agent":
                        # Final AIMessage from 'agent' node = the full response.
                        text = msg.content if isinstance(msg.content, str) else str(msg.content)
                        final_answer = text

                # ── Tool result ──────────────────────────────────────────────
                elif isinstance(msg, ToolMessage):
                    raw = str(msg.content) if msg.content else ""
                    preview = raw[:400] + " …" if len(raw) > 400 else raw
                    yield ("tool_result", msg.name, preview)

    # Emit final answer once after all streaming is done
    if final_answer:
        yield ("final_answer", final_answer)


# ---------------------------------------------------------------------------
# Simple blocking call
# ---------------------------------------------------------------------------

def run_normal_agent(user_query: str) -> str:
    """Run normal agent and return the final answer string."""
    final = ""
    for event in stream_normal_agent(user_query):
        if event[0] == "final_answer":
            final = event[1]
    return final or "I was unable to generate a response. Please try again."
