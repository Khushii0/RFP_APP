"""
Agent Router — single entry point that routes requests to the correct agent.

Usage:
    from src.agent_router import route_request, stream_route_request

    # Blocking call
    answer = route_request(user_query="...", deep_search=False)

    # Streaming call (for UI)
    for event in stream_route_request(user_query="...", deep_search=True):
        event_type = event[0]
        ...
"""

from src.deep_search_agent import run_deep_search_agent, stream_deep_search_agent
from src.normal_agent import run_normal_agent, stream_normal_agent


def route_request(user_query: str, deep_search: bool = False) -> str:
    """
    Route to the correct agent based on the deep_search flag.

    Args:
        user_query:  The user's input question or task.
        deep_search: If True, use the multi-step Deep Search agent.
                     If False (default), use the Normal ReAct agent.

    Returns:
        Final answer string from the selected agent.
    """
    if deep_search:
        return run_deep_search_agent(user_query=user_query)
    return run_normal_agent(user_query=user_query)


def stream_route_request(user_query: str, deep_search: bool = False):
    """
    Route to the correct agent and stream structured event tuples.

    Event tuple formats (shared by both agents):
      ("tool_call",   tool_name: str, args: dict)
      ("tool_result", tool_name: str, preview: str)
      ("reasoning",   text: str)
      ("final_answer",text: str)

    Additional events from Deep Search agent only:
      ("plan",        todos: list[str])
      ("subagent",    description: str)
      ("web_search",  query: str)

    Args:
        user_query:  The user's input question or task.
        deep_search: If True, stream from Deep Search agent.
                     If False, stream from Normal agent.

    Yields:
        Structured event tuples for the UI to render.
    """
    if deep_search:
        yield from stream_deep_search_agent(user_query=user_query)
    else:
        yield from stream_normal_agent(user_query=user_query)
