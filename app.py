"""
app.py — Gradio UI for the RFP Intelligence & Product Recommendation Agent.
Gradio 6.x compatible.

New in this version:
  - Live URL streaming in activity panel (sites being searched)
  - Per-tool timing badge (elapsed seconds)
  - Clean tool result formatting (no raw JSON)
  - Dedicated Sources section in activity panel
"""

import re
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

import gradio as gr
from src.agent_router import stream_route_request

from src.vector_store import ingest_rfp_chunks
ingest_rfp_chunks()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

body, .gradio-container { background: #0d1117 !important; color: #e6edf3 !important; }

.rfp-header {
    background: linear-gradient(135deg,#1a1f2e 0%,#0d1117 100%);
    border:1px solid #21262d; border-radius:12px;
    padding:20px 28px; margin-bottom:16px;
    display:flex; align-items:center; gap:14px;
}
.rfp-header-logo { font-size:34px; line-height:1; }
.rfp-header-text h1 { font-size:20px; font-weight:700; color:#58a6ff; margin:0 0 4px; }
.rfp-header-text p  { font-size:12px; color:#8b949e; margin:0; }

.panel-label {
    font-size:11px; font-weight:600; letter-spacing:.08em;
    text-transform:uppercase; color:#8b949e; margin-bottom:6px; padding-left:2px;
}

/* Chatbot */
.chatbot-wrap { background:#161b22 !important; border:1px solid #21262d !important; border-radius:10px !important; }

/* Input */
.query-box textarea {
    background:#161b22 !important; border:1px solid #30363d !important;
    color:#e6edf3 !important; border-radius:8px !important;
    font-size:14px !important; resize:none !important;
}
.query-box textarea:focus { border-color:#58a6ff !important; box-shadow:0 0 0 3px rgba(88,166,255,.15) !important; }
.query-box textarea::placeholder { color:#484f58 !important; }

/* Buttons */
.send-btn  { background:linear-gradient(135deg,#238636,#2ea043) !important; border:none !important; color:#fff !important; border-radius:8px !important; font-weight:600 !important; height:44px !important; transition:all .2s !important; }
.send-btn:hover  { filter:brightness(1.15) !important; transform:translateY(-1px) !important; }
.clear-btn { background:#21262d !important; border:1px solid #30363d !important; color:#8b949e !important; border-radius:8px !important; height:44px !important; transition:all .2s !important; }
.clear-btn:hover { border-color:#58a6ff !important; color:#58a6ff !important; }

/* Activity panel */
.activity-panel {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:14px; height:530px; overflow-y:auto; font-size:13px;
    scrollbar-width:thin; scrollbar-color:#30363d transparent;
}
.activity-panel::-webkit-scrollbar { width:5px; }
.activity-panel::-webkit-scrollbar-thumb { background:#30363d; border-radius:4px; }

/* Section headers inside activity */
.section-hdr {
    font-size:10px; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
    color:#484f58; border-bottom:1px solid #21262d; padding-bottom:5px;
    margin:10px 0 8px;
}
.section-hdr:first-child { margin-top:0; }

/* Events */
.ev { padding:7px 11px; border-radius:7px; margin-bottom:5px; animation:fadeIn .25s ease; line-height:1.5; }
@keyframes fadeIn { from{opacity:0;transform:translateY(3px)} to{opacity:1;transform:translateY(0)} }

.ev-plan     { background:#1a2035; border-left:3px solid #58a6ff; }
.ev-plan h4  { color:#58a6ff; margin:0 0 5px; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
.ev-plan ol  { margin:0; padding-left:18px; color:#c9d1d9; }
.ev-plan li  { margin-bottom:2px; font-size:12px; }

.ev-subagent { background:#1e1a2e; border-left:3px solid #c084fc; color:#d8b4fe; font-size:12px; }
.ev-subagent b { color:#c084fc; }

.ev-web  { background:#1a2535; border-left:3px solid #38bdf8; color:#7dd3fc; font-size:12px; }
.ev-web b { color:#38bdf8; }
.ev-web code { background:#0f2233; padding:1px 5px; border-radius:4px; }

.ev-tool { background:#1c2520; border-left:3px solid #3fb950; color:#85e89d; font-size:12px; display:flex; justify-content:space-between; align-items:flex-start; }
.ev-tool-left b { color:#3fb950; }
.ev-tool-left code { background:#0f1f10; padding:1px 5px; border-radius:4px; }
.timing { font-size:10px; color:#3fb950; background:#0f1f10; padding:1px 7px; border-radius:10px; white-space:nowrap; flex-shrink:0; margin-left:8px; }
.timing.pending { color:#484f58; background:#1a1a1a; }

.ev-result  { background:#1a1a1a; border-left:3px solid #484f58; color:#8b949e; font-size:11px; }
.ev-result b { color:#6e7681; }
.ev-result code { background:#0d1117; padding:1px 5px; border-radius:4px; }

.ev-reasoning { background:#1c2535; border-left:3px solid #f97316; color:#c9d1d9; font-size:12px; }
.ev-reasoning b { color:#f97316; }

.ev-done { background:#1c2d1c; border-left:3px solid #3fb950; color:#3fb950; font-weight:600; font-size:12px; }
.ev-idle { color:#484f58; font-size:12px; text-align:center; padding:24px; }

/* Source cards */
.source-card {
    background:#111827; border:1px solid #1e2a38; border-radius:7px;
    padding:7px 11px; margin-bottom:5px; font-size:12px;
    transition:border-color .2s;
}
.source-card:hover { border-color:#38bdf8; }
.source-num  { color:#38bdf8; font-weight:700; margin-right:6px; }
.source-title{ color:#c9d1d9; font-weight:500; }
.source-domain { color:#484f58; font-size:11px; }
.source-link { color:#58a6ff; font-size:11px; word-break:break-all; }
.source-link:hover { color:#79c0ff; }

/* URL visit row */
.ev-url { background:#111827; border-left:3px solid #1e3a5f; color:#8b949e; font-size:11px; padding:5px 10px; border-radius:5px; margin-bottom:4px; }
.ev-url a { color:#58a6ff; text-decoration:none; }
.ev-url a:hover { text-decoration:underline; }

/* Status bar */
.status-bar {
    display:flex; align-items:center; gap:8px; padding:6px 12px;
    border-radius:6px; background:#161b22; border:1px solid #21262d;
    font-size:12px; color:#8b949e; margin-bottom:8px;
}
.status-dot { width:8px; height:8px; border-radius:50%; background:#484f58; flex-shrink:0; }
.status-dot.active { background:#3fb950; animation:pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
"""

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def _extract_urls_from_tavily(text: str) -> list[str]:
    """Pull **URL:** https://... lines from a tavily result string."""
    return re.findall(r'\*\*URL:\*\*\s*(https?://\S+)', text)


def _extract_sources_from_answer(answer: str) -> list[dict]:
    """Parse [N] Title: URL lines from the final answer's ### Sources section."""
    sources = []
    # Match  [1] Some Title: https://...
    for m in re.finditer(r'\[(\d+)\]\s+([^:\n]+?):\s*(https?://\S+)', answer):
        sources.append({
            "num": m.group(1),
            "title": m.group(2).strip(),
            "url": m.group(3).strip(),
        })
    return sources


def _fmt_result(tool_name: str, raw: str) -> str:
    """Human-friendly one-liner for a tool result."""
    if tool_name in ("get_rfp_context", "get_product_context"):
        kb = round(len(raw.encode()) / 1024, 1)
        label = "RFP data" if "rfp" in tool_name else "product portfolio"
        return f"✓ {kb} KB of {label} loaded into context"
    if tool_name == "write_todos":
        n = raw.count("'content'") or raw.count('"content"')
        return f"✓ Research plan created ({n or '?'} tasks)"
    if tool_name == "tavily_search":
        urls = _extract_urls_from_tavily(raw)
        return f"✓ {len(urls)} web page(s) fetched"
    short = (raw[:160] + " …") if len(raw) > 160 else raw
    return short


# ---------------------------------------------------------------------------
# Activity HTML builder
# ---------------------------------------------------------------------------

def _build_activity_html(
    steps: list,          # list of (event_tuple, elapsed_sec | None)
    url_visits: list,     # list of str URLs (from tavily results)
    sources: list,        # list of {"num","title","url"}
    deep_search: bool,
    running: bool = False,
) -> str:

    if not steps:
        mode = "Deep Search" if deep_search else "Normal"
        dot = "🟢" if running else "⬜"
        return (
            f'<div class="ev-idle">{dot} Waiting for agent activity…<br>'
            f'<small style="color:#30363d;font-size:11px">Mode: {mode}</small></div>'
        )

    parts = []

    # ── STEPS section ──────────────────────────────────────────────────────
    parts.append('<div class="section-hdr">🔄 Agent Steps</div>')

    for ev, elapsed in steps:
        etype = ev[0]

        if etype == "plan":
            todos = ev[1]
            items = "".join(f"<li>{t}</li>" for t in todos)
            parts.append(
                f'<div class="ev ev-plan"><h4>📋 Research Plan</h4><ol>{items}</ol></div>'
            )

        elif etype == "subagent":
            parts.append(
                f'<div class="ev ev-subagent">🤖 <b>Sub-Agent:</b> {ev[1]}</div>'
            )

        elif etype == "web_search":
            parts.append(
                f'<div class="ev ev-web">🌐 <b>Searching:</b> <code>{ev[1]}</code></div>'
            )

        elif etype == "tool_call":
            name = ev[1]
            if name == "tavily_search":
                continue  # shown as web_search
            timing_html = '<span class="timing pending">timing…</span>'
            parts.append(
                f'<div class="ev ev-tool" id="tc-{name}">'
                f'<span class="ev-tool-left">🔧 <b>Tool:</b> <code>{name}</code></span>'
                f'{timing_html}</div>'
            )

        elif etype == "tool_result":
            name = ev[1]
            summary = _fmt_result(name, ev[2])
            t_badge = (
                f'<span class="timing">{elapsed:.2f}s</span>'
                if elapsed is not None
                else ""
            )
            parts.append(
                f'<div class="ev ev-result">'
                f'📄 <b>Result</b> <code>{name}</code>: {summary} {t_badge}'
                f'</div>'
            )

        elif etype == "tavily_result":
            # special: timing + summary for tavily
            name = "tavily_search"
            summary = ev[1]
            t_badge = (
                f'<span class="timing">{elapsed:.2f}s</span>'
                if elapsed is not None
                else ""
            )
            parts.append(
                f'<div class="ev ev-result">'
                f'📄 <b>Result</b> <code>{name}</code>: {summary} {t_badge}'
                f'</div>'
            )

        elif etype == "reasoning":
            text = ev[1]
            parts.append(
                f'<div class="ev ev-reasoning">💡 <b>Reasoning:</b> {text}</div>'
            )

        elif etype == "final_answer":
            parts.append('<div class="ev ev-done">✅ Answer generated</div>')

    # ── SITES VISITED section ──────────────────────────────────────────────
    if url_visits:
        parts.append('<div class="section-hdr">🌐 Sites Searched</div>')
        seen = set()
        for url in url_visits:
            if url in seen:
                continue
            seen.add(url)
            dom = _domain(url)
            short_url = url if len(url) <= 60 else url[:57] + "…"
            parts.append(
                f'<div class="ev-url">'
                f'<a href="{url}" target="_blank">🔗 {dom}</a>'
                f'<br><span style="color:#30363d">{short_url}</span>'
                f'</div>'
            )

    # ── SOURCES section ────────────────────────────────────────────────────
    if sources:
        parts.append('<div class="section-hdr">📚 Sources</div>')
        for s in sources:
            dom = _domain(s["url"])
            parts.append(
                f'<div class="source-card">'
                f'<span class="source-num">[{s["num"]}]</span>'
                f'<span class="source-title">{s["title"]}</span><br>'
                f'<span class="source-domain">{dom}</span> · '
                f'<a class="source-link" href="{s["url"]}" target="_blank">{s["url"][:70]}{"…" if len(s["url"])>70 else ""}</a>'
                f'</div>'
            )

    return "\n".join(parts)


def _status_html(running: bool, deep_search: bool) -> str:
    dot = "status-dot active" if running else "status-dot"
    mode = "🔴 Deep Search" if deep_search else "⚡ Normal"
    status = "Processing…" if running else "Ready"
    return (
        f'<div class="status-bar">'
        f'<span class="{dot}"></span><span>{status}</span>'
        f'<span style="margin-left:auto;color:#484f58">{mode}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Streaming handler
# ---------------------------------------------------------------------------

def respond(message: str, history: list, deep_search: bool):
    """Gradio 6.x streaming generator — yields (history, activity_html, status_html)."""
    if not message or not message.strip():
        yield history, _build_activity_html([], [], [], deep_search), _status_html(False, deep_search)
        return

    history = history + [{"role": "user", "content": message}]
    yield history, _build_activity_html([], [], [], deep_search, running=True), _status_html(True, deep_search)

    steps: list = []          # (event_tuple, elapsed | None)
    url_visits: list = []     # URLs fetched by tavily
    sources: list = []        # parsed from final answer
    final_answer = ""

    # Track tool-call start times  {tool_name: deque of timestamps}
    pending: dict = defaultdict(deque)

    for event in stream_route_request(message, deep_search=deep_search):
        now = time.perf_counter()
        etype = event[0]

        if etype == "tool_call":
            pending[event[1]].append(now)
            steps.append((event, None))

        elif etype == "tool_result":
            name = event[1]
            raw = event[2]
            elapsed = None
            if pending[name]:
                elapsed = now - pending[name].popleft()

            # If this is a tavily result, handle separately
            if name == "tavily_search":
                urls = _extract_urls_from_tavily(raw)
                url_visits.extend(urls)
                summary = f"✓ {len(urls)} web page(s) fetched"
                # Emit as tavily_result so we can show timing without duplicate tool card
                steps.append((("tavily_result", summary), elapsed))
            else:
                steps.append((event, elapsed))

        elif etype == "web_search":
            # Track start time for tavily
            pending["tavily_search"].append(now)
            steps.append((event, None))

        elif etype == "final_answer":
            final_answer = event[1]
            sources = _extract_sources_from_answer(final_answer)
            steps.append((event, None))

        else:
            steps.append((event, None))

        still_running = etype != "final_answer"
        activity = _build_activity_html(steps, url_visits, sources, deep_search, running=still_running)
        yield history, activity, _status_html(still_running, deep_search)

    bot = final_answer if final_answer else "⚠️ No response generated. Please check your API keys."
    history = history + [{"role": "assistant", "content": bot}]
    yield history, _build_activity_html(steps, url_visits, sources, deep_search, running=False), _status_html(False, deep_search)


def clear_all():
    return [], _build_activity_html([], [], [], False), _status_html(False, False)


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

HEADER = """
<div class="rfp-header">
  <div class="rfp-header-logo">🧠</div>
  <div class="rfp-header-text">
    <h1>RFP Intelligence &amp; Product Recommendation Agent</h1>
    <p>Newgen Software · GPT · Tavily Web Search · LangChain deepagents</p>
  </div>
</div>
"""

EXAMPLES = [
    "What are the key requirements in this RFP?",
    "Which Newgen products best cover the RFP requirements?",
    "What capability gaps exist between the RFP and Newgen's products?",
    "Recommend a new product idea based on the RFP gaps",
    "What integrations are required and does Newgen support them?",
]

with gr.Blocks(title="RFP Intelligence Agent") as demo:

    gr.HTML(HEADER)

    with gr.Row(equal_height=False):

        # ── LEFT: Chat ────────────────────────────────────────────────────
        with gr.Column(scale=6):
            gr.HTML('<div class="panel-label">💬 Chat</div>')

            chatbot = gr.Chatbot(
                value=[],
                height=480,
                show_label=False,
                elem_classes=["chatbot-wrap"],
                avatar_images=(None, "🧠"),
            )

            with gr.Row():
                query_box = gr.Textbox(
                    placeholder="Ask about RFP requirements, product gaps, recommendations…",
                    show_label=False,
                    lines=2,
                    max_lines=4,
                    elem_classes=["query-box"],
                    scale=8,
                )
                with gr.Column(scale=2, min_width=160):
                    send_btn  = gr.Button("Send ➤", elem_classes=["send-btn"], variant="primary")
                    clear_btn = gr.Button("🗑 Clear", elem_classes=["clear-btn"])

            deep_search_toggle = gr.Checkbox(
                label="🔴 Enable Deep Search Mode  (multi-step · web search · sub-agents)",
                value=False,
            )

            gr.Examples(examples=EXAMPLES, inputs=query_box, label="💡 Example queries")

        # ── RIGHT: Activity ───────────────────────────────────────────────
        with gr.Column(scale=4):
            gr.HTML('<div class="panel-label">⚡ Live Agent Activity</div>')
            status_box   = gr.HTML(value=_status_html(False, False))
            activity_box = gr.HTML(
                value=_build_activity_html([], [], [], False),
                elem_classes=["activity-panel"],
            )

    # ── Wiring ───────────────────────────────────────────────────────────

    def _submit(msg, hist, ds):
        yield from respond(msg, hist, ds)

    send_btn.click(
        fn=_submit,
        inputs=[query_box, chatbot, deep_search_toggle],
        outputs=[chatbot, activity_box, status_box],
        queue=True,
    ).then(fn=lambda: "", outputs=query_box)

    query_box.submit(
        fn=_submit,
        inputs=[query_box, chatbot, deep_search_toggle],
        outputs=[chatbot, activity_box, status_box],
        queue=True,
    ).then(fn=lambda: "", outputs=query_box)

    clear_btn.click(fn=clear_all, outputs=[chatbot, activity_box, status_box])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.queue(max_size=5).launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(),
    )
