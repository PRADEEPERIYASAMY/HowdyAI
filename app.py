"""
app.py  –  HowdyAI Streamlit frontend.
Premium Apple-style dark UI — fully consistent glassmorphism design.
"""

import os
import logging
import logging.config

import streamlit as st

from config import AppConfig
from main import build_pipeline_components, run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HowdyAI – TAMU Assistant",
    page_icon="🤠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Comprehensive Apple-style CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700&display=swap');

/* ────────────────────────────────────────────────
   1. GLOBAL RESET
   ──────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp, .stMarkdown, p, span, div {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Force full dark background everywhere */
.stApp, .stApp > div, section.main {
    background: #0d0d14 !important;
}

/* Remove default Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Chat container width */
.block-container {
    max-width: 860px !important;
    padding: 1.5rem 2rem 7rem !important;
    margin: 0 auto !important;
}

/* ────────────────────────────────────────────────
   2. SIDEBAR
   ──────────────────────────────────────────────── */
[data-testid="stSidebar"] > div:first-child {
    background: #111118 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}

/* All sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.85rem !important;
}

/* Sidebar section labels */
.sb-section {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.28) !important;
    margin: 1.2rem 0 0.5rem !important;
    display: block;
}

/* Sidebar buttons */
[data-testid="stSidebar"] button[kind="secondary"] {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    width: 100% !important;
    padding: 0.5rem 0.85rem !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    transition: background 0.18s, border-color 0.18s !important;
    box-shadow: none !important;
    margin-bottom: 0.4rem !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.11) !important;
    border-color: rgba(255,255,255,0.2) !important;
    color: white !important;
}
[data-testid="stSidebar"] button[kind="secondary"]:focus {
    box-shadow: 0 0 0 2px rgba(80,0,0,0.5) !important;
    border-color: rgba(80,0,0,0.6) !important;
    outline: none !important;
}

/* Sidebar metric */
[data-testid="stSidebar"] [data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    padding: 0.7rem 1rem !important;
}
[data-testid="stSidebar"] [data-testid="metric-container"] label {
    font-size: 0.7rem !important;
    color: rgba(255,255,255,0.35) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="stSidebar"] [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: white !important;
    line-height: 1.2 !important;
}

[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 1rem 0 !important;
}

/* Feature capability rows */
.feat-row {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 0.28rem 0;
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.82rem;
}
.feat-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: rgba(52,199,89,0.75);
    flex-shrink: 0;
}

/* ────────────────────────────────────────────────
   3. HERO HEADER
   ──────────────────────────────────────────────── */
.hero-wrap {
    text-align: center;
    padding: 2rem 1rem 1.8rem;
}
.hero-emoji {
    font-size: 3.2rem;
    display: block;
    margin-bottom: 0.55rem;
    animation: floatY 4s ease-in-out infinite;
}
@keyframes floatY {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-7px); }
}
.hero-title {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em !important;
    color: white !important;
    margin: 0 0 0.3rem !important;
    line-height: 1.1 !important;
}
.hero-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.38);
    font-weight: 400;
    margin: 0 0 0.8rem;
}
.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(52,199,89,0.12);
    border: 1px solid rgba(52,199,89,0.3);
    color: #34c759 !important;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 11px 3px 9px;
    border-radius: 20px;
}
.live-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #34c759;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(0.8); }
}

/* ────────────────────────────────────────────────
   4. CHAT MESSAGES
   ──────────────────────────────────────────────── */

/* Nuke all Streamlit default chat bubble styling */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    gap: 0 !important;
}
[data-testid="stChatMessage"] > div {
    background: transparent !important;
}

/* User bubble */
.bubble-user {
    background: linear-gradient(145deg, #6b0000, #500000);
    border-radius: 18px 18px 5px 18px;
    padding: 0.78rem 1.05rem;
    margin: 0.15rem 0 0.15rem 15%;
    color: rgba(255,255,255,0.93) !important;
    font-size: 0.93rem;
    font-weight: 400;
    line-height: 1.55;
    box-shadow: 0 2px 18px rgba(80,0,0,0.35);
    animation: fadeSlideR 0.22s ease;
    word-wrap: break-word;
}
@keyframes fadeSlideR {
    from { opacity: 0; transform: translateX(8px) translateY(4px); }
    to   { opacity: 1; transform: translateX(0) translateY(0); }
}

/* Assistant bubble */
.bubble-ai {
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px 18px 18px 5px;
    padding: 0.78rem 1.05rem;
    margin: 0.15rem 15% 0.15rem 0;
    color: rgba(255,255,255,0.88) !important;
    font-size: 0.93rem;
    font-weight: 400;
    line-height: 1.6;
    animation: fadeSlideL 0.22s ease;
    word-wrap: break-word;
}
@keyframes fadeSlideL {
    from { opacity: 0; transform: translateX(-8px) translateY(4px); }
    to   { opacity: 1; transform: translateX(0) translateY(0); }
}

/* Markdown inside bubbles */
.bubble-ai p, .bubble-user p { margin: 0 0 0.4em !important; }
.bubble-ai p:last-child, .bubble-user p:last-child { margin: 0 !important; }
.bubble-ai strong, .bubble-user strong { color: white !important; font-weight: 600 !important; }
.bubble-ai a { color: #60a5fa !important; text-decoration: underline; }

/* ────────────────────────────────────────────────
   5. META BADGES
   ──────────────────────────────────────────────── */
.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin: 0.35rem 0 0 0;
    padding: 0 2px;
    max-width: 100%;
    box-sizing: border-box;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    letter-spacing: 0.01em;
    border-width: 1px;
    border-style: solid;
}
.chip-cache   { background: rgba(52,199,89,0.12);  color: #34c759 !important; border-color: rgba(52,199,89,0.28); }
.chip-guard   { background: rgba(255,69,58,0.12);   color: #ff453a !important; border-color: rgba(255,69,58,0.28); }
.chip-latency { background: rgba(10,132,255,0.12);  color: #409cff !important; border-color: rgba(10,132,255,0.28); }
.chip-rewrite { background: rgba(191,90,242,0.12);  color: #da8fff !important; border-color: rgba(191,90,242,0.28); }

/* ────────────────────────────────────────────────
   6. SOURCE CITATIONS
   ──────────────────────────────────────────────── */
.sources-wrap {
    margin: 0.4rem 0 1.2rem 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 5px;
    padding: 0 2px;
    width: 100%;
    box-sizing: border-box;
}
.src-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: rgba(255,255,255,0.25);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-right: 3px;
    flex-shrink: 0;
    align-self: center;
}
.src-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    padding: 3px 9px;
    font-size: 0.74rem;
    font-weight: 400;
    color: rgba(255,255,255,0.6) !important;
    text-decoration: none !important;
    transition: background 0.16s, border-color 0.16s, color 0.16s, transform 0.16s;
    cursor: pointer;
    max-width: 200px;
    min-width: 0;
    box-sizing: border-box;
    /* No overflow:hidden here — use inner span instead */
}
/* Text inside chip gets the ellipsis, not the chip itself */
.src-chip .chip-text {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
    flex: 1;
}
.src-chip:hover {
    background: rgba(80,0,0,0.3) !important;
    border-color: rgba(140,0,0,0.5) !important;
    color: rgba(255,255,255,0.9) !important;
    transform: translateY(-1px);
}
.src-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(255,255,255,0.09);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.62rem;
    font-weight: 600;
    color: rgba(255,255,255,0.45) !important;
    flex-shrink: 0;
    line-height: 1.4;
}
.src-chip.src-local {
    border-style: dashed;
    border-color: rgba(255,255,255,0.07);
    opacity: 0.75;
}

/* ────────────────────────────────────────────────
   7. CHAT INPUT — Override Streamlit defaults fully
   ──────────────────────────────────────────────── */
/* Container */
[data-testid="stChatInput"] {
    background: rgba(28,28,38,0.95) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5), 0 1px 0 rgba(255,255,255,0.04) inset !important;
    backdrop-filter: blur(20px) !important;
    outline: none !important;
    transition: border-color 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(80,0,0,0.6) !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5), 0 0 0 3px rgba(80,0,0,0.15) !important;
}
/* Textarea */
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: rgba(255,255,255,0.9) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 400 !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    caret-color: #500000 !important;
    padding: 0.6rem 0.2rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.22) !important;
}
/* Send button */
[data-testid="stChatInput"] button {
    background: #500000 !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    transition: background 0.18s !important;
    box-shadow: none !important;
    outline: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    min-width: 36px !important;
    width: 36px !important;
    height: 36px !important;
    align-self: center !important;
    flex-shrink: 0 !important;
    margin-right: 4px !important;
}
[data-testid="stChatInput"] button svg {
    display: block !important;
    width: 16px !important;
    height: 16px !important;
    flex-shrink: 0 !important;
}
[data-testid="stChatInput"] button:hover {
    background: #6b0000 !important;
}
[data-testid="stChatInput"] button:focus {
    box-shadow: 0 0 0 2px rgba(80,0,0,0.5) !important;
}

/* Fix the sticky bottom bar that can clip the input right edge */
[data-testid="stBottom"] > div {
    background: rgba(13,13,20,0.92) !important;
    backdrop-filter: blur(16px) !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    padding: 0.75rem 1rem !important;
    box-sizing: border-box !important;
    max-width: 100% !important;
}

/* ────────────────────────────────────────────────
   8. MISC / POLISH
   ──────────────────────────────────────────────── */
/* Spinner */
.stSpinner > div { border-top-color: #500000 !important; }

/* Success / info toasts */
.stSuccess {
    background: rgba(52,199,89,0.1) !important;
    border: 1px solid rgba(52,199,89,0.25) !important;
    border-radius: 10px !important;
}
.stSuccess p { color: #34c759 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

/* Remove focus ring on sidebar items */
button:focus { outline: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline loader ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading HowdyAI…")
def load_pipeline():
    config = AppConfig(log_level="WARNING")
    logging.config.dictConfig(config.logging_config)
    comps = build_pipeline_components(config)
    return (config,) + comps


if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, meta?, user_query?}]

config, cache, memory, guardrail, rewriter, retriever, generator = load_pipeline()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:0.4rem 0 1.1rem;
                border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:1rem;">
        <span style="font-size:1.7rem;line-height:1">🤠</span>
        <div>
            <div style="font-size:1rem;font-weight:700;color:white;letter-spacing:-0.02em;">HowdyAI</div>
            <div style="font-size:0.68rem;color:rgba(255,255,255,0.35);text-transform:uppercase;
                        letter-spacing:0.07em;font-weight:500;">TAMU ASSISTANT</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Controls
    st.markdown('<span class="sb-section">Controls</span>', unsafe_allow_html=True)
    if st.button("🗑️  Clear Conversation"):
        st.session_state.messages = []
        memory.clear()
        st.success("Conversation cleared!")
    if st.button("🧹  Clear Response Cache"):
        cache.clear()
        st.success("Cache cleared!")

    # Stats
    st.markdown('<span class="sb-section">Stats</span>', unsafe_allow_html=True)
    stats = cache.stats()
    st.metric("Cached Responses", stats["entries"])

    # Capabilities
    st.markdown('<span class="sb-section">Capabilities</span>', unsafe_allow_html=True)
    for icon, feat in [
        ("🔀", "Hybrid retrieval + RRF"),
        ("✏️", "LLM query rewriting"),
        ("🧩", "Semantic chunking"),
        ("💬", "Multi-turn memory"),
        ("📎", "Inline citations"),
        ("🛡️", "Out-of-scope guardrail"),
        ("⚡", "Response caching"),
    ]:
        st.markdown(f"""
        <div class="feat-row">
            <span class="feat-dot"></span>
            <span>{icon}&nbsp;{feat}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="color:rgba(255,255,255,0.18);font-size:0.68rem;line-height:1.6;">'
                'Built for CSCE 670<br>Texas A&M University</p>', unsafe_allow_html=True)


# ── Hero ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
    <span class="hero-emoji">🤠</span>
    <h1 class="hero-title">HowdyAI</h1>
    <p class="hero-sub">Your AI-powered Texas A&amp;M University assistant</p>
    <span class="live-pill">
        <span class="live-dot"></span>Live &nbsp;·&nbsp; tamu.edu
    </span>
</div>""", unsafe_allow_html=True)


# ── Helper renderers ────────────────────────────────────────────────────────────

def render_bubbles_from_history():
    """Render the full stored message history as styled HTML bubbles."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="bubble-ai">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            _render_meta(msg.get("meta", {}), msg.get("user_query", ""))


def _render_meta(meta: dict, user_query: str = ""):
    """Render badge row + source chips for one assistant turn."""
    badges = []
    if meta.get("cache_hit"):
        badges.append('<span class="chip chip-cache">⚡ Cache hit</span>')
    if meta.get("guardrail_hit"):
        badges.append('<span class="chip chip-guard">🛡 Out of scope</span>')
    if meta.get("latency_s") is not None:
        badges.append(f'<span class="chip chip-latency">⏱ {meta["latency_s"]}s</span>')
    if meta.get("rewritten_query") and meta["rewritten_query"] != user_query:
        rq = meta["rewritten_query"][:60]
        badges.append(f'<span class="chip chip-rewrite" title="{rq}">✏️ Rewritten</span>')
    if badges:
        st.markdown(
            f'<div class="meta-row">{"".join(badges)}</div>',
            unsafe_allow_html=True,
        )

    sources = meta.get("sources", [])
    if sources:
        chips = '<span class="src-label">Sources</span>'
        for s in sources:
            url   = s.get("url", "")
            title = s.get("title", url)
            idx   = s.get("index", "?")
            short = (title[:40] + "…") if len(title) > 40 else title
            if url.startswith("http"):
                chips += (
                    f'<a class="src-chip" href="{url}" target="_blank">'
                    f'<span class="src-num">{idx}</span>'
                    f'<span class="chip-text">{short}</span></a>'
                )
            else:
                base  = os.path.basename(url) if url else title
                short_b = (base[:40] + "…") if len(base) > 40 else base
                chips += (
                    f'<span class="src-chip src-local" title="{url}">'
                    f'<span class="src-num">{idx}</span>'
                    f'<span class="chip-text">{short_b}</span></span>'
                )
        st.markdown(
            f'<div class="sources-wrap">{chips}</div>',
            unsafe_allow_html=True,
        )


# ── Render history, then input ─────────────────────────────────────────────────

render_bubbles_from_history()

if user_input := st.chat_input("Ask anything about Texas A&M…  Howdy! 🤠"):
    # Immediately render user bubble
    st.markdown(
        f'<div class="bubble-user">{user_input}</div>',
        unsafe_allow_html=True,
    )
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Run pipeline
    with st.spinner("Searching and thinking…"):
        result = run_pipeline(
            query=user_input,
            config=config,
            cache=cache,
            memory=memory,
            guardrail=guardrail,
            rewriter=rewriter,
            retriever=retriever,
            generator=generator,
        )

    answer  = result["answer"]
    sources = result["sources"]
    meta = {
        "cache_hit":       result["cache_hit"],
        "guardrail_hit":   result["guardrail_hit"],
        "latency_s":       result["latency_s"],
        "rewritten_query": result["rewritten_query"],
        "sources":         sources,
    }

    # Render AI bubble + meta
    st.markdown(f'<div class="bubble-ai">{answer}</div>', unsafe_allow_html=True)
    _render_meta(meta, user_input)

    # Persist
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "meta":       meta,
        "user_query": user_input,
    })
