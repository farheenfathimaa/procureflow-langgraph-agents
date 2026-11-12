# ProcureFlow — Multi-Agent Procurement Assistant

ProcureFlow is a portfolio-grade, free-tier-friendly procurement assistant built with **LangGraph**, **FastAPI**, **hybrid RAG**, **MCP**, **Google OAuth 2.0**, and **human-in-the-loop approval**.

A natural-language purchase request such as *“Buy 20 laptops, budget 30k USD”* is decomposed by an orchestrator and handed to three specialist agents:

- **SupplierAgent** compares suppliers using retrieved catalogue data, total cost, warranty, lead time, certifications and rating.
- **PolicyAgent** checks approval thresholds, security, sustainability and tender rules against retrieved policy documents.
- **BudgetAgent** reads a Google Sheets budget ledger (or a clearly labelled local demo ledger) and checks availability.

The graph then pauses at a checkpointed **approval gate** when the amount is over the delegated limit, confidence is low, policy requires review, the budget is unverified/over budget, a supplier is missing, or an agent fails. Approval or rejection resumes the same workflow from PostgreSQL rather than restarting it.

---

## Contents

- [Architecture](#architecture)
- [Feature highlights](#feature-highlights)
- [Quickstart with Docker Compose](#quickstart-with-docker-compose)
- [Local development without Docker](#local-development-without-docker)
- [Configuration](#configuration)
- [API walkthrough](#api-walkthrough)
- [Human-in-the-loop approval](#human-in-the-loop-approval)
- [Hybrid RAG and citations](#hybrid-rag-and-citations)
- [Google OAuth 2.0 and Sheets](#google-oauth-20-and-sheets)
- [MCP supplier tools](#mcp-supplier-tools)
- [Evaluation](#evaluation)
- [Reliability and safety](#reliability-and-safety)
- [Testing and CI](#testing-and-ci)
- [Design decisions](#design-decisions)
- [Repository layout](#repository-layout)
- [Limitations and next steps](#limitations-and-next-steps)
- [License](#license)

---

## Architecture

```mermaid
flowchart LR
    U[Client / curl / React] -->|POST /api/v1/requests| API[FastAPI]
    API --> RT[AppRuntime]
    RT --> G[LangGraph workflow]

    subgraph G[LangGraph orchestrator]
        GR[Guardrail] --> OR[Orchestrator]
        OR --> R{Router}
        R --> PA[PolicyAgent]
        R --> SA[SupplierAgent]
        R --> BA[BudgetAgent]
        PA --> RISK[Risk gate]
        SA --> RISK
        BA --> RISK
        RISK -->|needs approval| HITL[Approval interrupt]
        RISK -->|safe| FIN[Finalize]
        HITL -->|resume approve/reject| FIN
    end

    PA --> HR[Hybrid retriever]
    SA --> HR
    HR --> BM[BM25 index]
    HR --> VEC[(PostgreSQL + pgvector)]
    HR --> RR[Local cross-encoder reranker]

    BA --> OA[Google OAuth manager]
    OA --> GS[Google Sheets API]
    BA --> DEMO[Local demo ledger]

    G --> CP[(Postgres checkpointer)]
    API --> MEM[(Redis cache / pending set)]
    G -. optional traces .-> LS[LangSmith]

    subgraph M[MCP server]
        M1[search_suppliers]
        M2[get_supplier]
    end
    M -. reads .-> CORPUS[(Synthetic supplier corpus)]
```

### Request lifecycle

```mermaid
sequenceDiagram
    participant C as Client
    participant A as FastAPI
    participant W as LangGraph
    participant P as Policy/Supplier/Budget
    participant H as Human reviewer
    participant D as Postgres

    C->>A: POST /api/v1/requests
    A->>W: invoke(initial state)
    W->>W: guardrail + orchestrate
    W->>P: sequential specialist handoffs
    P-->>W: structured results + citations
    W->>D: checkpoint state, then interrupt
    W-->>C: 201 awaiting_approval + interrupt payload
    H->>A: POST /api/v1/approvals/{thread_id}/decision
    A->>W: Command(resume=decision)
    W->>D: load checkpoint, continue
    W->>W: finalize recommendation
    W-->>A: approved / rejected
    A-->>H: final recommendation with citations
```

The state is a typed `ProcurementState` (`app/state.py`); every agent output crosses a Pydantic boundary before it is stored as JSON so a Postgres checkpoint can always be resumed.

---

## Feature highlights

| Area | Implementation |
| --- | --- |
| Orchestration | LangGraph `StateGraph` with explicit guardrail → orchestrator → router → specialists → risk → approval/finalize edges |
| Agent outputs | Pydantic structured outputs (`OrchestratorOutput`, `PolicyDraft`, `SupplierDraft`, `Recommendation`, …) |
| LLM providers | LiteLLM with Groq primary, Gemini fallback, bounded exponential backoff, Redis response cache |
| Safe failure | Per-agent failure records, deterministic parser fallback, conservative approval gate, max-iteration guard |
| RAG | 20 synthetic supplier/policy documents, local sentence-transformers embeddings, pgvector, BM25, reciprocal rank fusion, local cross-encoder, citations |
| Persistence | LangGraph Postgres checkpointer; Redis short-term memory and pending-approval index; in-memory fallback |
| HITL | LangGraph `interrupt()` + `Command(resume=...)`; approve/reject endpoints; decisions persist in the checkpoint |
| OAuth | Google Authorization Code flow with one-time CSRF state, offline access, refresh, 0600 token storage, reauthorization errors |
| Google Sheets | Configurable range → key/value budget parser with retries and explicit failure (never a silent fallback) |
| MCP | Supplier lookup exposed as MCP tools over stdio or Streamable HTTP |
| Evaluation | 27 questions, RAGAS metrics, measured results written to `docs/eval_results.md` |
| Ops | Dockerfile, Compose, health endpoint, Docker health check, Makefile, GitHub Actions, pytest/ruff |
| Cost | Free LLM tiers + local embeddings/reranker; no paid hosting required for the demo |

---

## Quickstart with Docker Compose

### Prerequisites

- Docker Engine 24+ with Compose v2
- Free API keys from [Groq](https://console.groq.com/) and optionally [Google AI Studio](https://aistudio.google.com/)
- Optional: Google OAuth client credentials and a Google Sheet for live budget data

### 1. Configure

```bash
git clone https://github.com/farheenfathimaa/procureflow-langgraph-agents.git
cd procureflow-langgraph-agents
cp .env.example .env
```

Edit `.env` and set at least:

```dotenv
GROQ_API_KEY=your-groq-key
GEMINI_API_KEY=your-gemini-key          # optional fallback
POSTGRES_PASSWORD=choose-a-password
```

The Compose app resource builds `DATABASE_URL` from `POSTGRES_*` and points Redis at the Compose service. Local development outside Compose should use `localhost` instead of `postgres`/`redis`.

### 2. Start the stack

```bash
docker compose up --build
```

Services:

| Service | URL / port | Notes |
| --- | --- | --- |
| API | http://localhost:8000 | Swagger at `/docs` |
| Postgres + pgvector | `localhost:5432` | Checkpoints + vector documents |
| Redis | `localhost:6379` | Cache and pending-approval set |
| MCP | `http://localhost:8100/mcp` | Streamable HTTP transport |

The first start downloads the local embedding and reranker models into the `model_cache` volume, so it can take several minutes. The API health check has a 180-second start period for this reason.

### 3. Verify

```bash
curl http://localhost:8000/health
```

`status` is `ok` when Postgres/Redis/vector documents/LLM configuration are all healthy; it is `degraded` when an optional dependency is unavailable. The Docker health check still treats an HTTP 200 response as “process is up”.

### 4. Submit a request

```bash
curl -s -X POST http://localhost:8000/api/v1/requests \
  -H 'Content-Type: application/json' \
  -d '{"request":"Buy 20 laptops, budget 30k USD","requester":"demo@company.com"}'
```

A high-value request returns `201` with `status: "awaiting_approval"` and an `interrupt` payload. Approve it with:

```bash
curl -s -X POST http://localhost:8000/api/v1/requests/<thread_id>/approve \
  -H 'Content-Type: application/json' \
  -d '{"approver":"CFO","comment":"Approved for Q3"}'
```

---

## Local development without Docker

Prerequisites: Python 3.11, [`uv`](https://docs.astral.sh/uv/), and optionally PostgreSQL 16 with pgvector plus Redis.

```bash
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,embeddings,eval]"
```

For a fully local, dependency-free demo, use the in-memory checkpointer and hash embeddings:

```bash
export MOCK_LLM=true
export CHECKPOINTER_BACKEND=memory
export DATABASE_URL=""
export REDIS_URL=""
export EMBEDDING_BACKEND=hash
export RERANKER_ENABLED=false
uvicorn app.main:app --reload --port 8000
```

For realistic retrieval with local models but no Postgres:

```bash
export CHECKPOINTER_BACKEND=memory
export DATABASE_URL=""
export REDIS_URL=""
export EMBEDDING_BACKEND=sentence-transformers
export RERANKER_ENABLED=true
export MOCK_LLM=false
uvicorn app.main:app --reload --port 8000
```

To use Postgres locally, start it with pgvector, set `CHECKPOINTER_BACKEND=postgres` and a `DATABASE_URL` such as `postgresql://procureflow:procureflow@localhost:5432/procureflow`. The app creates the `vector` extension, `rag_documents` table/indexes, and LangGraph checkpoint tables automatically, then seeds the bundled corpus.

Useful Make targets:

```bash
make install   # create .venv and install dev/embedding/eval extras
make up        # docker compose up --build
make test      # offline pytest (MOCK_LLM=true)
make lint      # ruff check
make format    # ruff format
make ingest    # re-embed the corpus inside the app container
make eval      # run RAGAS and write docs/eval_results.md
make mcp       # run the MCP supplier server over stdio
```

---

## Configuration

All settings are environment variables, loaded through `app/config.py`. `.env.example` is the canonical reference.

### Core

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOCK_LLM` | `false` | Deterministic structured outputs; tests/CI never call external APIs |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | empty | Provider credentials |
| `GROQ_MODEL` | `groq/openai/gpt-oss-120b` | Primary model |
| `GEMINI_MODEL` | `gemini/gemini-3.5-flash` | Fallback model |
| `LLM_MAX_RETRIES` | `2` | Extra attempts per provider |
| `CHECKPOINTER_BACKEND` | `postgres` | `postgres` or `memory` |
| `DATABASE_URL` | local Postgres URL | Checkpoints + pgvector |
| `REDIS_URL` | local Redis URL | LLM cache + short-term memory; empty disables Redis |
| `APPROVAL_THRESHOLD_USD` | `10000` | Amount that forces human review |
| `CONFIDENCE_THRESHOLD` | `0.75` | Minimum acceptable agent confidence |
| `MAX_GRAPH_ITERATIONS` | `16` | Bounded graph iteration guard |

### RAG

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_BACKEND` | `sentence-transformers` | `sentence-transformers` or deterministic `hash` |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `EMBEDDING_DIM` | `384` | pgvector dimension |
| `RERANKER_ENABLED` | `true` | Enable local cross-encoder |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local reranker |
| `RAG_TOP_K` / `RAG_CANDIDATE_K` | `4` / `12` | Final contexts / candidate depth |
| `RRF_K` | `60` | Reciprocal rank fusion constant |
| `AUTO_INGEST` | `true` | Seed the bundled corpus on startup |

### Google and tracing

| Variable | Purpose |
| --- | --- |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | OAuth client credentials |
| `GOOGLE_REDIRECT_URI` | Must match the Google console (default `http://localhost:8000/auth/google/callback`) |
| `GOOGLE_SHEET_ID` / `GOOGLE_SHEET_RANGE` | Budget spreadsheet and range |
| `GOOGLE_TOKEN_PATH` | Token file (default `/data/google_token.json`, volume-backed, mode `0600`) |
| `LOCAL_TOTAL_BUDGET_USD` / `LOCAL_COMMITTED_USD` | Demo ledger used only when Sheets is not configured |
| `LANGSMITH_ENABLED` | Opt-in tracing; off by default |
| `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | LangSmith credentials/project; no key needed while disabled |

Never commit `.env`, OAuth client-secret JSON files, or token files. They are covered by `.gitignore` and `.dockerignore`.

---

## API walkthrough

Swagger UI: `http://localhost:8000/docs`.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/requests` | Start a workflow |
| `GET` | `/api/v1/requests/{thread_id}` | Read checkpointed state |
| `GET` | `/api/v1/approvals` | List pending approvals |
| `POST` | `/api/v1/approvals/{thread_id}/decision` | Canonical approve/reject endpoint |
| `POST` | `/api/v1/requests/{thread_id}/approve` | Convenience approve endpoint |
| `POST` | `/api/v1/requests/{thread_id}/reject` | Convenience reject endpoint |
| `GET` | `/auth/google` | Start OAuth Authorization Code flow |
| `GET` | `/auth/google/callback` | OAuth callback (code exchange) |
| `GET` | `/auth/google/status` | Non-secret connection status |
| `GET` | `/auth/google/budget` | Read and parse the budget sheet |
| `POST` | `/auth/google/disconnect` | Delete the local token |
| `GET` | `/health` | Dependency and configuration health |
| `GET` | `/` | Service metadata |

### Create

```bash
curl -s -X POST http://localhost:8000/api/v1/requests \
  -H 'Content-Type: application/json' \
  -d '{"request":"Buy 20 laptops, budget 30k USD","requester":"demo@company.com","currency":"USD"}'
```

Relevant response fields:

```json
{
  "thread_id": "…",
  "status": "awaiting_approval",
  "parsed_request": {"item": "laptops", "quantity": 20, "total_amount_usd": 30000},
  "policy": {"requires_approval": true, "citations": ["…"]},
  "suppliers": {"recommended_supplier": "Orbit Systems", "candidates": ["…"]},
  "budget": {"source": "local_demo", "within_budget": true},
  "risk": {"approval_required": true, "reasons": ["…"]},
  "interrupt": {"amount_usd": 30000, "reasons": ["…"]},
  "can_resume": true
}
```

Guardrail-blocked requests still return `201` but with `status: "blocked"`, `can_resume: false`, and no supplier/policy execution.

### Decide

```bash
curl -s -X POST http://localhost:8000/api/v1/approvals/<thread_id>/decision \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approve","approver":"CFO","comment":"Within plan"}'
```

`decision` must be `approve` or `reject`; `approver` is required. Invalid decisions return `422`. Deciding on a workflow that is not waiting returns `409`. Unknown threads return `404`.

### List pending

```bash
curl -s http://localhost:8000/api/v1/approvals
```

The pending set is stored in Redis (with an in-process fallback); the definitive state always comes from the checkpointer.

---

## Human-in-the-loop approval

The risk gate runs deterministically after every specialist handoff. Approval is required when **any** of these is true:

1. Total amount exceeds `APPROVAL_THRESHOLD_USD`.
2. Minimum agent confidence is below `CONFIDENCE_THRESHOLD`.
3. Policy explicitly requires approval or reports a violation.
4. Budget is over the available amount or could not be verified.
5. No suitable supplier is recommended.
6. Any agent/guardrail error was recorded.
7. The purchase amount is missing.

The approval node calls `interrupt(payload)` and checkpoints before pausing. The payload contains the request summary, amount, confidence, reasons, recommended supplier/total, policy rationale, available budget, and requester.

On resume, the API validates the decision against Pydantic, removes the thread from the pending set, and continues the same checkpoint. A rejection produces a final `rejected` recommendation rather than discarding the analysis. Repeating a decision is rejected with `409` instead of silently overwriting the prior human decision.

### Safe failure and loop bounds

- Each agent catches its own exceptions, stores a bounded error record, and returns a conservative result.
- The orchestrator falls back to a deterministic regex parser if the LLM is unavailable.
- LLM calls are bounded: per-provider retries with exponential backoff, then fallback to the next provider, then a typed `LLMUnavailableError`.
- The router stops at `MAX_GRAPH_ITERATIONS` and forces human review.
- User input is guardrailed before any LLM call; retrieved documents are delimited as untrusted reference data and the prompts explicitly forbid following instructions inside them.

---

## Hybrid RAG and citations

The bundled corpus (`rag/data/documents.jsonl`) contains **20 fictional documents**: 12 supplier catalogues and 8 procurement policies (`POL-001` … `POL-008`). No real supplier, employee or customer data is used.

Retrieval flow per query:

1. **Dense channel** — local `sentence-transformers` embedding → cosine similarity in pgvector (or an in-memory store for tests).
2. **Sparse channel** — BM25 over the same corpus.
3. **Fusion** — reciprocal rank fusion: `Σ 1 / (RRF_K + rank)` across channels.
4. **Re-ranking** — local cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) reorders fused candidates.
5. **Citations** — only retrieved document IDs may become citations; hallucinated IDs are dropped, and each result carries the document title, source, type, snippet and score.

SupplierAgent additionally narrows candidates to the parsed category so a laptop request cannot be recommended a monitor because of a lexical quirk. Volume discounts are applied only when the document states both the percentage and the qualifying quantity.

The corpus is committed, so startup does not require a generation step. `python -m rag.data.generate_documents` regenerates it; `python -m rag.ingest --force` re-embeds it into Postgres.

---

## Google OAuth 2.0 and Sheets

ProcureFlow implements the **Authorization Code** flow directly with `google-auth-oauthlib`:

1. `GET /auth/google` creates a one-time, high-entropy `state`, stores it in Redis/memory with a 10-minute TTL, and redirects to Google with `access_type=offline` and `prompt=consent`.
2. Google redirects to `GET /auth/google/callback?code=…&state=…`.
3. The state is validated and deleted **before** code exchange to prevent replay.
4. Tokens are written atomically to `GOOGLE_TOKEN_PATH` with directory mode `0700` and file mode `0600`.
5. `credentials()` refreshes expired access tokens and rewrites the token file. A rejected refresh token deletes the local token and asks the user to reconnect. Transient refresh errors surface as retryable connector errors.
6. `GET /auth/google/budget` reads the configured range and parses rows.

Expected sheet layout (headers optional):

| A | B |
| --- | --- |
| Total budget | `$120,000` |
| Committed | `$42,000` |
| Available | `$78,000` |

Accepted aliases include Annual Budget/Budget/Total, Spent/Committed/Used/Expenses and Available/Remaining/Balance. `$`, commas and accounting parentheses are supported. If a sheet is configured but unreadable, the BudgetAgent reports `within_budget: null` and the request goes to human review — it does **not** silently substitute demo data.

The default OAuth scope is `https://www.googleapis.com/auth/spreadsheets.readonly`. The redirect URI in Google Cloud Console must exactly match `GOOGLE_REDIRECT_URI`.

---

## MCP supplier tools

`mcp_server/server.py` exposes the catalogue through the official MCP SDK (`MCPServer`):

- `search_suppliers(query, category=None, max_results=3)` — BM25 supplier search with category filtering and citation payloads.
- `get_supplier(supplier_id)` — lookup by catalogue ID (for example `SUP-LAP-001`).

Run over stdio (Claude Desktop/Inspector):

```bash
python -m mcp_server.server
```

Run over Streamable HTTP (used by Compose):

```bash
MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8100 python -m mcp_server.server
```

The endpoint is `http://localhost:8100/mcp`. Tool responses include `supplier_id`, commercial metadata, a summary, and a citation object; unknown IDs raise a tool error rather than returning fabricated data.

---

## Evaluation

The evaluation set has **27 questions** across approval thresholds, budget rules, tenders, security, sustainability, ethics, emergency purchasing and supplier facts (`evals/eval_questions.jsonl`). Each item has a ground truth and relevant document IDs.

```bash
make eval
# or: python -m evals.run_eval
```

The harness:

- uses the same hybrid retriever as the application (in-memory store, so no Docker/Postgres needed),
- generates answers with Groq and records citations,
- scores **faithfulness**, **answer relevancy**, **context precision** and **context recall** with RAGAS,
- uses a separate free-tier Groq judge,
- writes only metrics that were actually returned to [`docs/eval_results.md`](docs/eval_results.md), including per-metric sample coverage.

Measured results from the latest full run are published in [`docs/eval_results.md`](docs/eval_results.md). Numbers are intentionally not hard-coded in this README; the report is generated by the harness. The free Groq tier applies per-minute and per-day token limits, so individual judge calls can fail; the report records how many samples contributed, and failed calls are never replaced with invented scores. A 20-document synthetic corpus with 27 questions is a pipeline smoke test, not a claim of production retrieval quality.

Useful flags:

```bash
python -m evals.run_eval --limit 3 --output /tmp/eval.md --raw /tmp/eval.json
```

`--raw` is optional and gitignored. LangSmith tracing is controlled by `LANGSMITH_ENABLED`; when false, no tracing key is required or sent.

---

## Reliability and safety

| Risk | Mitigation |
| --- | --- |
| Provider outage / rate limit | Per-provider retries + exponential backoff + Gemini fallback + bounded failure |
| Structured-output drift | JSON response format + Pydantic validation + JSON extraction from fenced output |
| Prompt injection | Unicode normalisation, zero-width stripping, deterministic pattern guard, untrusted-context delimiters |
| Hallucinated citations | Citation IDs are resolved against actual retrieval hits; unknown IDs are dropped |
| Fabricated suppliers/prices | Supplier prompts and mock path rely on retrieved documents only; no fit → low confidence → review |
| Silent budget fallback | Configured Sheets errors become unverified budget + HITL, never demo data |
| Infinite agent loop | `MAX_GRAPH_ITERATIONS`, explicit router guard, LangGraph recursion limit |
| Lost approvals | Postgres checkpointer is durable and resumable by thread ID; Redis pending index has an in-process fallback (single process) |
| Secret leakage | `.env`/tokens/client-secret JSON gitignored; tokens stored `0600`; API returns no secrets; logs omit keys |
| OAuth CSRF/replay | One-time state with TTL, deleted before exchange |
| Repeated human decisions | Non-waiting workflows return `409`; prior decision remains in checkpoint |

No authentication/authorization is implemented for API callers in this demo. Put the API behind an identity-aware proxy and add reviewer SSO/roles before any real deployment.

---

## Testing and CI

The suite runs fully offline:

```bash
MOCK_LLM=true .venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
```

Coverage includes:

- guardrails and prompt-injection variants,
- deterministic parsing and corpus/eval integrity,
- BM25, hybrid retrieval, RRF and citations,
- LLM mock/no-network mode, JSON fences, retry counts and provider fallback,
- every specialist agent, safe-failure paths and the risk gate,
- workflow checkpointing, approval, rejection, blocking, 404/409 semantics and currency assumptions,
- FastAPI endpoints, validation, OAuth unconfigured paths and OpenAPI,
- Google token storage, OAuth state/CSRF and code exchange, budget parsing, retries and provider selection,
- MCP tool registration and payloads.

GitHub Actions (`.github/workflows/ci.yml`) installs with `uv sync --extra dev`, runs Ruff and pytest with `MOCK_LLM=true`, hash embeddings, memory checkpointer, and empty database/Redis URLs. Tests make **zero external API calls**.

---

## Design decisions

- **LangGraph 0.6 / LangChain 0.3** was chosen because it is the stable stack compatible with RAGAS 0.2.15. Newer RAGAS releases conflicted with newer LangChain community packages in this environment.
- **Plain JSON state** at checkpoints, with Pydantic validation at agent boundaries. This keeps Postgres serialisation predictable and makes the API view schema explicit.
- **Sequential specialist handoffs** rather than parallel fan-out: each result is visible to the next agent and the graph is easier to reason about, evaluate and resume. The router is a single place to change ordering or add agents.
- **Deterministic risk gate** instead of asking an LLM whether approval is needed. This is auditable, testable and impossible to prompt away.
- **Google Sheets as the budget source of truth when configured**, with a clearly labelled demo ledger only when Sheets is absent. Connector errors are not hidden.
- **MCP for supplier lookup** so the same catalogue can be consumed by other agents or desktop clients without duplicating business logic.
- **Local embeddings and reranker** to keep the demo free and private. Hash embeddings exist only for deterministic tests.
- **Postgres checkpointer** rather than a custom session store, so approval resume, history and restart behaviour are handled by LangGraph.
- **Evaluation is measured, not asserted.** Failed metrics are reported as “not measured” rather than invented.

---

## Repository layout

```text
app/                 FastAPI app, config, state, workflow, LLM client, DB, memory, schemas
agents/              Orchestrator, policy, supplier, budget, risk, shared base
rag/                 Corpus generator/data, loaders, embeddings, BM25, pgvector, RRF, reranker, ingest
connectors/          Google OAuth manager/token store and Sheets budget connector
mcp_server/          MCP supplier server and tools
evals/               27-question dataset and RAGAS harness
tests/               Offline unit/integration tests
docs/                Generated evaluation report
docker/init.sql      Enables pgvector before app startup
Dockerfile           Python 3.11 CPU-only image
docker-compose.yml   Postgres+pgvector, Redis, API and MCP
Makefile             install/up/test/lint/format/ingest/eval/mcp
pyproject.toml       Dependencies, extras, Ruff/pytest config
.env.example         Configuration reference
```

---

## Limitations and next steps

- **The full Compose stack was not started end-to-end in this environment.** The Docker CLI was unavailable here (Podman 5.7 only), so the Compose file is provided and reviewed but was not launched as a single stack. Its two stateful dependencies were run separately with Podman: real PostgreSQL 16 + pgvector and Redis, against which corpus ingestion, pgvector retrieval, checkpointed start, approval, resume and reload-from-a-fresh-runtime all passed. The API container build itself remains unexercised.
- **Google Sheets/OAuth live flow is not completed here** because no spreadsheet ID or browser consent/refresh token was available. The OAuth state/CSRF/token-exchange logic and Sheets parsing/retry logic are unit-tested with fakes.
- **Postgres checkpointer + pgvector were verified against a real database**, but only in single-node Podman runs with the hash embedder for speed. The sentence-transformers path was verified against the in-memory store, not against pgvector with 384-dimension vectors; the schema and dimension are configurable and tested, but a full-size embedding ingest is left to the Compose quickstart.
- **No frontend.** A minimal React approval page was an optional stretch goal and is not included; Swagger and curl are the review interface.
- **No caller authentication.** Add SSO, reviewer roles, audit-log export and rate limiting before production.
- **Approval threshold logic is USD-only.** Non-USD requests are recorded with an explicit assumption error and routed for review.
- **Synthetic corpus.** Supplier/policy facts are fictional and should be replaced with governed, versioned enterprise sources and a freshness policy.
- **Token storage is local-file.** Production should use KMS/secret manager encryption and key rotation.
- **Single graph process assumption.** The Redis pending set is shared, but the in-process fallback is not; run one API replica or use Redis in multi-replica deployments.
- **Eval size is intentionally small.** Expand with adversarial, multi-hop and “no-answer” questions, and track retrieval recall separately from generation quality.
- **The RAGAS judge is a free-tier model.** Per-minute/per-day token limits can reject individual judge calls, so a score is the mean over the calls that succeeded. The report records coverage and never substitutes fabricated scores; run with paid/self-hosted judging capacity for a fully populated 27-question report.

---

## License

MIT — see [`LICENSE`](LICENSE).
