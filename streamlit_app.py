"""
streamlit_app.py — Trupti Dance Academy site + RAG assistant (split layout).

Run:
    streamlit run streamlit_app.py

Needs (same folder):
    tda_pipeline.py
    trupti_dance_academy_corpus.txt
    trupti_recital_corpus_2026.txt
    .env  with  OPENAI_API_KEY=<your Nebius key>
"""

import uuid
import os
import streamlit as st

# --------------------------------------------------------------------------- #
# Credentials & observability.
# Bridge Streamlit Cloud "Secrets" -> environment variables, so both the
# pipeline (reads os.environ["OPENAI_API_KEY"]) and LangSmith auto-tracing
# (reads LANGSMITH_* env vars) work in the cloud AND locally via .env.
# This MUST run before `import tda_pipeline`.
# --------------------------------------------------------------------------- #
def _bridge_secrets_to_env():
    keys = ["OPENAI_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING",
            "LANGSMITH_PROJECT", "LANGSMITH_ENDPOINT"]
    try:
        for k in keys:
            if k not in os.environ and k in st.secrets:
                os.environ[k] = str(st.secrets[k])
    except Exception:
        # No secrets.toml locally — fine, .env / load_dotenv covers local runs.
        pass


_bridge_secrets_to_env()

import tda_pipeline

st.set_page_config(page_title="Trupti Dance Academy", page_icon="✦", layout="wide")

# --------------------------------------------------------------------------- #
# Styling — clean, minimal, modern. Typography-forward, restrained palette.
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --ink: #18181b;
    --muted: #6b7280;
    --line: #ececf0;
    --card: #f8f8fa;
    --accent: #a4133c;     /* restrained deep rose */
    --bg: #ffffff;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1180px; }
#MainMenu, footer, header { visibility: hidden; }

.tda-brand { font-size: .82rem; letter-spacing: .22em; text-transform: uppercase;
    color: var(--accent); font-weight: 600; margin-bottom: .4rem; }
.tda-hero-h { font-size: 3.1rem; line-height: 1.05; font-weight: 700;
    letter-spacing: -.02em; margin: 0 0 1rem 0; }
.tda-hero-h .em { color: var(--accent); }
.tda-hero-p { font-size: 1.12rem; color: var(--muted); max-width: 30rem;
    line-height: 1.6; margin-bottom: 1.6rem; }

.tda-section-label { font-size: .78rem; letter-spacing: .18em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin: 2.6rem 0 1rem 0; }
.tda-h2 { font-size: 1.7rem; font-weight: 700; letter-spacing: -.01em; margin: 0 0 1rem 0; }

.tda-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: .9rem; }
.tda-card { background: var(--card); border: 1px solid var(--line); border-radius: 16px;
    padding: 1.15rem 1.25rem; }
.tda-card h3 { font-size: 1.02rem; font-weight: 600; margin: 0 0 .25rem 0; }
.tda-card p { font-size: .9rem; color: var(--muted); margin: 0; line-height: 1.5; }

.tda-recital { background: linear-gradient(135deg, #1f1320 0%, #3a1228 100%);
    color: #fff; border-radius: 20px; padding: 2rem 2.2rem; margin-top: .5rem; }
.tda-recital .k { font-size: .78rem; letter-spacing: .18em; text-transform: uppercase;
    color: #f3b4c8; font-weight: 600; }
.tda-recital h2 { font-size: 1.9rem; font-weight: 700; margin: .5rem 0 .6rem 0; }
.tda-recital p { color: #e7d7df; line-height: 1.6; margin: 0; max-width: 34rem; }
.tda-stats { display: flex; gap: 2.4rem; margin-top: 1.4rem; }
.tda-stats .n { font-size: 1.7rem; font-weight: 700; }
.tda-stats .l { font-size: .8rem; color: #d9b9c6; }

.tda-about p { color: var(--muted); line-height: 1.7; font-size: 1.0rem; max-width: 38rem; }
.tda-foot { color: var(--muted); font-size: .85rem; border-top: 1px solid var(--line);
    margin-top: 2.6rem; padding-top: 1.2rem; }

/* chat rail */
.tda-chat-head { font-size: 1.15rem; font-weight: 700; margin-bottom: .2rem; }
.tda-chat-sub { font-size: .85rem; color: var(--muted); line-height: 1.5; margin-bottom: .8rem; }
.tda-chip { display: inline-block; }
div[data-testid="stChatMessage"] { background: var(--card); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Pipeline — built once, cached across reruns.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Warming up the studio assistant…")
def get_app():
    return tda_pipeline.build_app()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex
if "pending" not in st.session_state:
    st.session_state.pending = None


def submit_question(q: str):
    st.session_state.pending = q


# --------------------------------------------------------------------------- #
# Layout: site (left, wider) | assistant (right)
# --------------------------------------------------------------------------- #
left, right = st.columns([1.7, 1], gap="large")

# ---------------------------- LEFT: the website ---------------------------- #
with left:
    st.markdown('<div class="tda-brand">Trupti Dance Academy · Melissa, TX</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="tda-hero-h">Where every age finds its <span class="em">rhythm</span>.</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="tda-hero-p">Bollywood and BollyX dance for kids, teens, and adults — '
        'energetic classes, a welcoming community, and an annual recital to shine on stage.</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="tda-section-label">Programs</div>', unsafe_allow_html=True)
    st.markdown('<div class="tda-h2">Classes for every age</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tda-grid">
      <div class="tda-card"><h3>Parent &amp; Child</h3><p>Ages 2–3 · move and bond together</p></div>
      <div class="tda-card"><h3>Kids</h3><p>Ages 4–6 and 7–8 · playful foundations</p></div>
      <div class="tda-card"><h3>Girls Group</h3><p>Ages 9–15 · technique and choreography</p></div>
      <div class="tda-card"><h3>Teen Dance</h3><p>Ages 12–18 · Fridays 5–6 PM</p></div>
      <div class="tda-card"><h3>Ladies Group</h3><p>Ages 15+ · Bollywood for all levels</p></div>
      <div class="tda-card"><h3>BollyX Fitness</h3><p>Adults · dance-cardio workout</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Ask the assistant on the right for exact schedules, pricing, and how to enroll.")

    st.markdown('<div class="tda-section-label">On stage</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tda-recital">
      <div class="k">Our First Annual Recital</div>
      <h2>Rhythm on stage — May 30, 2026</h2>
      <p>Our very first recital brought the whole studio together for an evening of group and
      solo performances, from our littlest dancers to our ladies group — to a full, cheering house.</p>
      <div class="tda-stats">
        <div><div class="n">60+</div><div class="l">Performers</div></div>
        <div><div class="n">5</div><div class="l">Age categories</div></div>
        <div><div class="n">6–8 PM</div><div class="l">An unforgettable evening</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="tda-section-label">About</div>', unsafe_allow_html=True)
    st.markdown('<div class="tda-h2">Dance, fitness, and fun in Melissa</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="tda-about"><p>Trupti Dance Academy is a Bollywood and BollyX studio in '
        'Melissa, Texas, welcoming dancers of every age and ability. Whether your little one is '
        'taking their first steps or you are joining the ladies group for the joy of it, there is '
        'a place for you here.</p></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="tda-foot">Trupti Dance Academy · Melissa, TX · '
        '<a href="https://truptidance.com" target="_blank">truptidance.com</a> · '
        'Questions? Ask the assistant — it speaks English, Hindi, and Telugu.</div>',
        unsafe_allow_html=True)

# ---------------------------- RIGHT: the chatbot --------------------------- #
with right:
    st.markdown('<div class="tda-chat-head">Studio Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="tda-chat-sub">Ask about classes, schedules, pricing, policies, or the '
        'recital — in English, Hindi, or Telugu.</div>', unsafe_allow_html=True)

    # starter suggestions
    if not st.session_state.messages:
        s1, s2 = st.columns(2)
        s1.button("Class schedule & pricing", use_container_width=True,
                  on_click=submit_question, args=("What classes do you offer and how much do they cost?",))
        s2.button("When was the recital?", use_container_width=True,
                  on_click=submit_question, args=("When was the recital and who performed?",))
        s3, s4 = st.columns(2)
        s3.button("How do I enroll?", use_container_width=True,
                  on_click=submit_question, args=("How do I enroll my child in a class?",))
        s4.button("Refund policy", use_container_width=True,
                  on_click=submit_question, args=("What is your refund policy?",))

    # history
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar=("✦" if m["role"] == "assistant" else None)):
            st.markdown(m["content"])

    # input form (text_input + button works inside a column; st.chat_input does not)
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([4, 1])
        user_text = col_in.text_input("Message", label_visibility="collapsed",
                                      placeholder="Type your question…")
        sent = col_btn.form_submit_button("Send", use_container_width=True)
    if sent and user_text.strip():
        submit_question(user_text.strip())

    # process a pending question (from form or a suggestion button)
    if st.session_state.pending:
        q = st.session_state.pending
        st.session_state.pending = None
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant", avatar="✦"):
            with st.spinner("Thinking…"):
                try:
                    app = get_app()
                    result = tda_pipeline.answer(app, q, st.session_state.thread_id)
                    reply = result["final_answer"] or \
                        "Sorry, I didn't catch that — could you rephrase?"
                except Exception as e:
                    reply = ("The assistant isn't available right now. Please check that the "
                             "Nebius key and corpus files are in place.")
                    with st.expander("Debug detail"):
                        st.exception(e)
                st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()
