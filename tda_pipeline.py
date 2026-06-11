"""
tda_pipeline.py — Trupti Dance Academy RAG pipeline, importable.

Exposes:
    build_app()                     -> compiled LangGraph (build once, cache in Streamlit)
    answer(app, question, thread_id) -> {"final_answer", "route", "language"}

Needs in the working directory:
    tda_corpus.txt                 (classes)
    tda_recital_corpus_2026.txt    (recital)
    .env  with  OPENAI_API_KEY=<your Nebius key>
"""

import os
import re
import json
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

NEBIUS_BASE = "https://api.tokenfactory.nebius.com/v1/"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
CHAT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

CORPORA = [
    {"path": "tda_corpus.txt",
     "metadata": {"doc_type": "classes", "year": 2026, "event_status": "current"}},
    {"path": "tda_recital_corpus_2026.txt",
     "metadata": {"doc_type": "recital", "year": 2026, "event_status": "completed"}},
]

_DIVIDER_LINE = re.compile(r"^[\s=\-_*~]+$")


# --------------------------------------------------------------------------- #
# Ingest helpers
# --------------------------------------------------------------------------- #
def _strip_indexing_note(text: str) -> str:
    idx = text.find("\n=== ")
    return text[idx + 1:] if idx != -1 else text


def _clean_chunk(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _DIVIDER_LINE.match(ln)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _load_chunks() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100,
        separators=["\n=== ", "\n\n", "\n", " ", ""],
    )
    chunks: list[Document] = []
    for c in CORPORA:
        path = Path(c["path"])
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path.resolve()}")
        raw = _strip_indexing_note(path.read_text(encoding="utf-8"))
        parent = Document(page_content=raw, metadata=c["metadata"])
        for ch in splitter.split_documents([parent]):
            ch.page_content = _clean_chunk(ch.page_content)
            if ch.page_content:
                chunks.append(ch)
    return chunks


# --------------------------------------------------------------------------- #
# Build the whole graph. Nodes are closures so they capture llm + retrievers
# without leaking module-level globals.
# --------------------------------------------------------------------------- #
def build_app():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY (your Nebius key) is not set in the environment / .env")

    all_chunks = _load_chunks()

    embeddings = OpenAIEmbeddings(
        base_url=NEBIUS_BASE, model=EMBED_MODEL,
        check_embedding_ctx_length=False, api_key=api_key,
    )

    def make_hybrid(doc_type: str, k: int = 4) -> EnsembleRetriever:
        # One store per shelf, built from that shelf's chunks only — so no
        # metadata filter is needed, and there's no chromadb dependency.
        subset = [d for d in all_chunks if d.metadata["doc_type"] == doc_type]
        dense = InMemoryVectorStore.from_documents(
            subset, embedding=embeddings).as_retriever(search_kwargs={"k": k})
        sparse = BM25Retriever.from_documents(subset)
        sparse.k = k
        return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])

    retrievers = {"classes": make_hybrid("classes"), "recital": make_hybrid("recital")}

    llm = ChatOpenAI(base_url=NEBIUS_BASE, model=CHAT_MODEL,
                     temperature=0, api_key=api_key)

    # ---- prompts ---------------------------------------------------------- #
    ROUTER_SYSTEM = """You are the router for the Trupti Dance Academy support bot.
Classify the user's question into EXACTLY ONE category:
- classes: regular weekly dance classes — enrollment, schedule, timings, pricing,
  the class refund/cancellation policy, class age groups, attire, prior experience.
- recital: the academy's annual recital — recital date, who performed, recital
  registration, recital participation or costume fees, the recital refund/withdrawal
  policy, performer categories, recital tickets.
- out_of_scope: anything the class and recital documents would not cover — private
  one-on-one lessons, other studios, unrelated topics.
Rules:
- If a question mentions the recital, costumes, performing, or the show, choose recital.
- If a refund/fee/cancellation question does NOT mention the recital, choose classes.
- If it clearly isn't about classes or the recital, choose out_of_scope.
Respond with ONLY the category word: classes, recital, or out_of_scope."""

    DETECT_SYSTEM = """Detect the language of the user's message and translate it to English.
Return ONLY a JSON object (no markdown, no extra text) with keys:
  "detected_language": language name in English (e.g. "English", "Hindi", "Telugu")
  "script": "latin" if written in roman/latin letters, else the native script name
  "english_text": the message translated to English (copy as-is if already English)"""

    GENERATE_SYSTEM = """You are a friendly support assistant for Trupti Dance Academy.
Answer the QUESTION using ONLY the CONTEXT.
- If the context fully supports an answer, reply clearly and concisely.
- If the context does NOT contain the answer, reply with exactly: NOT_IN_DOCS
Never invent prices, dates, policies, or details not present in the context."""

    ESCALATION_MSG = (
        "I couldn't find that in our information, but I'd be happy to connect you with the "
        "Trupti Dance Academy team, who can help you directly. They'll follow up shortly."
    )

    VALID = {"classes", "recital", "out_of_scope"}

    def _extract_json(raw: str) -> dict:
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            try:
                return json.loads(m.group(0)) if m else {}
            except Exception:
                return {}

    def classify(question: str) -> str:
        resp = llm.invoke([("system", ROUTER_SYSTEM), ("human", question)])
        raw = resp.content.strip().lower()
        for label in VALID:
            if label in raw:
                return label
        return "classes"

    # ---- nodes ------------------------------------------------------------ #
    def detect_translate(state: dict) -> dict:
        resp = llm.invoke([("system", DETECT_SYSTEM), ("human", state["text"])])
        data = _extract_json(resp.content)
        return {
            "language": data.get("detected_language", "English"),
            "script": data.get("script", "latin"),
            "english_text": data.get("english_text", state["text"]),
        }

    def router(state: dict) -> dict:
        return {"route": classify(state.get("english_text") or state["text"])}

    def route_picker(state: dict) -> str:
        return state["route"]

    def retrieve(state: dict) -> dict:
        return {"context": retrievers[state["route"]].invoke(state["english_text"])}

    def generate(state: dict) -> dict:
        context = "\n\n---\n\n".join(d.page_content for d in state["context"])
        user = f"CONTEXT:\n{context}\n\nQUESTION: {state['english_text']}"
        resp = llm.invoke([("system", GENERATE_SYSTEM), ("human", user)])
        return {"draft": resp.content.strip()}

    def needs_escalation(state: dict) -> str:
        return "escalate" if "NOT_IN_DOCS" in state["draft"] else "translate_back"

    def escalate(state: dict) -> dict:
        return {"draft": ESCALATION_MSG}

    def translate_back(state: dict) -> dict:
        if state["language"].lower().startswith("english"):
            return {"final_answer": state["draft"]}
        resp = llm.invoke([
            ("system", f"Translate the message into {state['language']}, keeping a natural, "
                       f"friendly tone. Preserve every fact. Return only the translation."),
            ("human", state["draft"]),
        ])
        return {"final_answer": resp.content.strip()}

    # ---- assemble --------------------------------------------------------- #
    class State(TypedDict):
        text: str
        language: str
        script: str
        english_text: str
        route: str
        context: list
        draft: str
        final_answer: str

    g = StateGraph(State)
    for name, fn in [
        ("detect_translate", detect_translate), ("router", router),
        ("retrieve", retrieve), ("generate", generate),
        ("escalate", escalate), ("translate_back", translate_back),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("detect_translate")
    g.add_edge("detect_translate", "router")
    g.add_conditional_edges("router", route_picker,
                            {"classes": "retrieve", "recital": "retrieve",
                             "out_of_scope": "escalate"})
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", needs_escalation,
                            {"escalate": "escalate", "translate_back": "translate_back"})
    g.add_edge("escalate", "translate_back")
    g.add_edge("translate_back", END)

    return g.compile(checkpointer=MemorySaver())


def answer(app, question: str, thread_id: str) -> dict:
    """Run one question through the graph. Returns final_answer / route / language."""
    cfg = {"configurable": {"thread_id": thread_id}}
    out = app.invoke({"text": question}, cfg)
    return {
        "final_answer": out.get("final_answer", ""),
        "route": out.get("route", "out_of_scope"),
        "language": out.get("language", "English"),
    }
