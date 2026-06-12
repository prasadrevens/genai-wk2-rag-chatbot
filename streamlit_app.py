"""
streamlit_app.py — Trupti Dance Academy support chatbot (redesigned UI).

Dark-navy / cyan terminal aesthetic. Wraps the RAG pipeline in tda_pipeline.py
(build_app + answer) with per-session conversational memory.

Run:
    streamlit run streamlit_app.py

Secrets (Streamlit Cloud) or .env (local):
    OPENAI_API_KEY (Nebius key), LANGSMITH_API_KEY, LANGSMITH_TRACING, LANGSMITH_PROJECT
"""

import os
import uuid

import streamlit as st

# --------------------------------------------------------------------------- #
# Secrets -> env  (MUST run before importing the pipeline so build_app sees keys)
# Local dev uses .env (loaded inside tda_pipeline); on Cloud, bridge st.secrets.
# --------------------------------------------------------------------------- #
for _key in ("OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING",
             "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"):
    try:
        if _key in st.secrets and _key not in os.environ:
            os.environ[_key] = str(st.secrets[_key])
    except Exception:
        pass  # no secrets.toml locally — .env handles it

from tda_pipeline import build_app, answer  # noqa: E402

# --------------------------------------------------------------------------- #
# Real Trupti Dance Academy content (replaces the mockup's placeholder text).
# --------------------------------------------------------------------------- #
ACADEMY = "Dance Academy"
CLASSES = ["Kids Dance (4–7)", "Kids Dance (7–12)", "Teen Dance (12–18)",
           "Ladies Dance (18+)", "BollyX Fitness (16+)"]
LOCATION = "Melissa Community Center · Melissa, TX"
LANGUAGES = "English · हिन्दी"

GREETING = ("Hi 👋 I'm the Dance Academy assistant. Ask me about classes, "
            "schedules, fees, policies, or the recital — in English, Hindi, or Telugu.")

SUGGESTIONS = [
    "What classes do you offer?",
    "How much are classes?",
    "When is the recital?",
]

st.set_page_config(page_title=f"{ACADEMY} — Assistant",
                   page_icon="🩰", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Styling — dark navy + cyan, JetBrains Mono (display/labels) + Inter (body).
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --bg:#0A0E27; --panel:#0F1533; --panel-2:#131B3D;
  --border:#1E2A52; --border-cyan:#2BD4E8;
  --accent:#5EE7FB; --accent-dim:#22D3EE;
  --text:#E6EAF5; --muted:#8B93B8; --faint:#5A6291;
  --mono:'JetBrains Mono',ui-monospace,monospace;
  --sans:'Inter',-apple-system,system-ui,sans-serif;
}
.stApp{background:var(--bg);}
#MainMenu,header,footer{visibility:hidden;}
.block-container{padding-top:2.2rem;max-width:880px;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:var(--bg);border-right:1px solid var(--border);}
section[data-testid="stSidebar"] .block-container{padding-top:1.6rem;}
.brand{font-family:var(--mono);font-weight:700;font-size:1.45rem;letter-spacing:-.5px;color:var(--text);}
.brand .accent{color:var(--accent);}
.tagline{font-family:var(--mono);font-size:.72rem;color:var(--faint);margin:.15rem 0 1.2rem;letter-spacing:.5px;}
.rule{height:1px;background:var(--border);margin:0 0 1.3rem;}
.card{border:1px solid var(--border);border-radius:12px;padding:.9rem 1rem;margin-bottom:.85rem;background:var(--panel);}
.card-h{font-family:var(--mono);font-size:.72rem;font-weight:600;color:var(--accent);letter-spacing:1px;margin-bottom:.5rem;}
.card-b{font-family:var(--sans);font-size:.86rem;color:var(--text);line-height:1.6;}
.card-b .dim{color:var(--muted);}
.copyright{font-family:var(--mono);font-size:.68rem;color:var(--faint);margin-top:1.1rem;}

/* ---------- hero ---------- */
.hero{font-family:var(--mono);font-weight:700;font-size:2.55rem;line-height:1.1;text-align:center;color:var(--text);letter-spacing:-1px;margin:.2rem 0 .1rem;}
.hero .accent{color:var(--accent);}
.sub{font-family:var(--mono);font-size:.92rem;color:var(--muted);text-align:center;margin:0 auto 1.4rem;max-width:600px;line-height:1.5;}

/* ---------- suggestion chips (st.button) ---------- */
div[data-testid="stHorizontalBlock"] .stButton>button{
  width:100%;font-family:var(--mono);font-size:.82rem;font-weight:500;
  color:var(--accent);background:transparent;border:1px solid var(--border-cyan);
  border-radius:999px;padding:.5rem .9rem;transition:all .15s ease;}
div[data-testid="stHorizontalBlock"] .stButton>button:hover{
  background:rgba(94,231,251,.10);border-color:var(--accent);color:var(--accent);}

/* ---------- chat panel ---------- */
.panel-top{border:1px solid var(--border-cyan);border-bottom:none;
  border-radius:16px 16px 0 0;background:var(--panel);padding:1.1rem 1.3rem .4rem;
  box-shadow:0 0 24px rgba(43,212,232,.07);}
.panel-title{font-family:var(--mono);font-size:.78rem;font-weight:700;color:var(--accent);letter-spacing:1.5px;}
.panel-divider{height:1px;background:var(--border);margin:.7rem 0 .2rem;}

/* chat bubbles via st.chat_message */
div[data-testid="stChatMessage"]{background:transparent;padding:.35rem 0;}
.stChatMessage [data-testid="stChatMessageContent"]{font-family:var(--sans);}
/* assistant bubble */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"]{
  background:var(--panel-2);border:1px solid var(--border);border-radius:14px;
  padding:.8rem 1rem;color:var(--text);}
/* user bubble */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  background:var(--accent);border-radius:14px;padding:.8rem 1rem;color:#06243a;font-weight:500;}

/* meta caption under assistant answers */
.meta{font-family:var(--mono);font-size:.66rem;color:var(--faint);margin:.1rem 0 .3rem 3rem;letter-spacing:.5px;}

/* ---------- input ---------- */
div[data-testid="stChatInput"]{border:1px solid var(--border-cyan);border-radius:14px;background:var(--panel);}
div[data-testid="stChatInput"] textarea{font-family:var(--sans);color:var(--text);}
div[data-testid="stChatInput"] textarea::placeholder{color:var(--faint);}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# App + session state
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Warming up the studio… (embedding corpus)")
def get_app():
    return build_app()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"web-{uuid.uuid4()}"   # stable -> memory works
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": GREETING, "meta": None}]

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f'<div class="brand">🩰 Dance<span class="accent">Academy</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="tagline">// where Bollywood comes alive</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="card-h">◇ CLASSES</div>'
        '<div class="card-b">' + '<br>'.join(CLASSES) + '</div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><div class="card-h">◇ WHERE WE DANCE</div>'
        f'<div class="card-b">{LOCATION}<br><span class="dim">All classes $60 / 4-class session</span></div></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="card"><div class="card-h">◇ ASK IN ANY LANGUAGE</div>'
        f'<div class="card-b">{LANGUAGES}<br><span class="dim">The assistant replies in your language</span></div></div>',
        unsafe_allow_html=True)

    if st.button("Start a new chat ↻", use_container_width=True):
        st.session_state.thread_id = f"web-{uuid.uuid4()}"
        st.session_state.messages = [{"role": "assistant", "content": GREETING, "meta": None}]
        st.rerun()

    st.markdown(f'<div class="copyright">© 2026 {ACADEMY}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown('<div class="hero">A Dance Academy<br>That <span class="accent">Never Sleeps</span></div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub">Ask our AI assistant about classes, fees, schedules, '
            'and the recital — answers, instantly.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Suggestion chips -> queue a question
# --------------------------------------------------------------------------- #
pending = None
cols = st.columns(len(SUGGESTIONS))
for col, text in zip(cols, SUGGESTIONS):
    if col.button(text, key=f"sug-{text}"):
        pending = text

# --------------------------------------------------------------------------- #
# Chat panel header + history
# --------------------------------------------------------------------------- #
st.markdown('<div class="panel-top"><div class="panel-title">DANCE ACADEMY / ASSISTANT.</div>'
            '<div class="panel-divider"></div></div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = "🩰" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("meta"):
            st.markdown(f'<div class="meta">{msg["meta"]}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
typed = st.chat_input("Ask me anything about Dance Academy…")
if typed:
    pending = typed

# --------------------------------------------------------------------------- #
# Handle a question
# --------------------------------------------------------------------------- #
if pending:
    st.session_state.messages.append({"role": "user", "content": pending, "meta": None})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(pending)

    with st.chat_message("assistant", avatar="🩰"):
        with st.spinner("Thinking…"):
            try:
                app = get_app()
                result = answer(app, pending, st.session_state.thread_id)
                reply = result.get("final_answer") or "Sorry, I didn't catch that — could you rephrase?"
                meta = f"route: {result.get('route','—')}  ·  language: {result.get('language','—')}"
            except Exception as e:
                reply = ("Something went wrong reaching the assistant. Check that "
                         "`OPENAI_API_KEY` is set, then try again.")
                meta = f"error: {type(e).__name__}"
        st.markdown(reply)
        st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})
    st.rerun()
