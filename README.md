#Dance Academy — RAG Chatbot Assistant
[View HTML Report](docs/index.html)

A multilingual customer-support assistant for [Dance Academy]
(a Bollywood / BollyX studio in Melissa, TX), built as a Streamlit website with the chatbot
on the side. Parents and students can ask about **enrollment, schedule, pricing, policies, and
the annual recital** — in **English, Hindi ** — and the bot escalates to a human when
the answer isn't in the documents.

---

## What's under the hood

A retrieval-augmented generation (RAG) pipeline orchestrated as a **LangGraph** state machine:

```
detect_translate → router → retrieve → generate → (escalate?) → translate_back
```

| Concept | Where it lives |
|---|---|
| **Translate-first multilingual** | `detect_translate` → answer in English → `translate_back` |
| **Metadata routing** | `router` classifies `classes` / `recital` / `out_of_scope` |
| **Per-shelf hybrid retrieval** | In-memory vector store (dense) + BM25 (sparse) via `EnsembleRetriever`, one store per `doc_type` |
| **Grounding guard** | `generate` emits `NOT_IN_DOCS` → softened human hand-off |
| **Conversation memory** | LangGraph `MemorySaver` checkpointer, keyed by session `thread_id` |
| **Observability** | LangSmith auto-tracing (config-only, no code) |

LLM + embeddings run through **Nebius** (OpenAI-compatible API): `Llama-3.3-70B-Instruct`
for chat, `Qwen3-Embedding-8B` for embeddings.

---

## Project files

```
streamlit_app.py                 # entry point: website + chat UI
tda_pipeline.py                  # the RAG pipeline (importable)
tda_corpus.txt                   # classes corpus
tda_recital_corpus_2026.txt      # recital corpus
requirements.txt                 # pinned dependencies
.gitignore                       # keeps secrets out of git
```

---

## Run locally

1. Clone and enter the repo:
   ```bash
   git clone https://github.com/<your-username>/ -dance-rag.git
   cd  -dance-rag
   ```
2. Install dependencies (a virtual environment is recommended):
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root (this file is git-ignored — never commit it):
   ```
   OPENAI_API_KEY=your-nebius-key
   LANGSMITH_API_KEY=your-langsmith-key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT= -rag
   ```
4. Run:
   ```bash
   streamlit run streamlit_app.py
   ```

The first launch is slow (~10–20s) while embeddings build; after that it's cached.

---

## Deploy on Streamlit Community Cloud

### 1. Push to GitHub

With all files in one folder:

```bash
git init
git add .
git commit -m "  Dance Academy RAG app"
git branch -M main
git remote add origin https://github.com/<your-username>/ -dance-rag.git
git push -u origin main
```

Before pushing, run `git status` and confirm **`.env` is NOT listed**.

### 2. Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub (authorize Streamlit on first use).
2. Click **Create app** (upper-right).
3. Fill in: repository, branch `main`, main file `streamlit_app.py`. Optionally set a custom subdomain.
4. Open **Advanced settings**, set **Python 3.12**, and paste your secrets (TOML):
   ```toml
   OPENAI_API_KEY = "your-nebius-key"
   LANGSMITH_API_KEY = "your-langsmith-key"
   LANGSMITH_TRACING = "true"
   LANGSMITH_PROJECT = " -rag"
   ```
5. Click **Deploy**.

Updating later: just `git push` to `main` — changes redeploy automatically.

> Secrets are read from the environment. The app bridges Streamlit Cloud **Secrets** into
> environment variables at startup, so the same code runs locally (via `.env`) and in the cloud.

---

## Observability with LangSmith

Tracing is **configuration, not code**. With `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`
in the environment, LangChain and LangGraph auto-instrument every run. In your
[LangSmith](https://smith.langchain.com) ` -rag` project you'll see, per question:

- each graph node as a span (`detect_translate` → `router` → `retrieve` → `generate` → `translate_back`)
- the router's classification decision
- the chunks retrieved on each call (great for debugging a wrong answer)
- latency and token/cost per LLM call

Tracing happens at the LangChain layer, so it captures everything even though calls route
through Nebius rather than OpenAI.

---

## Notes & limits

- **Free-tier memory (~1 GB):** the corpus is tiny and dense retrieval uses langchain-core's
  in-memory vector store (no chromadb), so memory stays light and boot is fast.
- **Memory is half-built by design:** the checkpointer persists per-conversation state, but true
  follow-up handling ("and the recital fee?") needs a `condense_question` step — the next planned
  enhancement.
- **Corpus `[VERIFY]` fields:** exact venue, fees, and refund windows in the recital corpus are
  placeholders until confirmed.

---

## Roadmap

- `condense_question` — history-aware follow-ups (makes memory conversational)
- Re-ranking — fetch top-20, cross-encoder re-score to top-4
- RAGAS evaluation — measure faithfulness / relevance against the 95% targets
- Graph RAG — Neo4j over performer categories and schedule relationships
- Channels — Gmail (loop-guarded) and WhatsApp adapters over the same pipeline
