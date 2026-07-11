# SHL Assessment Recommender

A conversational AI agent that recommends SHL psychometric assessments for hiring scenarios. Given a job description or natural-language description of a role, it asks clarifying questions and returns a grounded shortlist of assessments — with links to the SHL product catalog.

**Live demo:** https://shl-recommender-wtwd.onrender.com
> Hosted on Render's free tier — first load may take **2–3 minutes** to spin up.

---

## How it works

```
User message
  │
  ├─ 1. Build retrieval query from conversation history (last N user turns)
  ├─ 2. Domain pinning — always inject flagship items (OPQ32r, Verify G+, etc.)
  │      when signals in the query suggest they're relevant
  ├─ 3. Hybrid search: FAISS semantic (70%) + BM25 keyword (30%) → top candidates
  ├─ 4. Inject candidates as grounded context into LLM system prompt
  ├─ 5. Single Groq API call (Llama 3.3 70B) with conversation history
  ├─ 6. Parse JSON response
  └─ 7. Validate every recommended URL against catalog before returning
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **Stateless API** | Caller sends full history each turn — no server-side session state, trivially scalable |
| **Hybrid retrieval** | Semantic search handles paraphrased queries; keyword covers exact product names |
| **FAISS flat index** | Exact search is correct for 377 items — no approximation needed |
| **Domain pinning** | Retrieval is probabilistic; flagship items get pinned when domain signals are present so they're never silently dropped from context |
| **Single LLM call per turn** | Avoids multi-step pipeline complexity; stays within timeout budget |
| **URL validation layer** | Every recommendation is cross-checked against the catalog — hallucinated URLs are rejected before they reach the caller |
| **HF Inference API fallback** | If local sentence-transformers aren't available (RAM-constrained deploy), embeddings fall back to HuggingFace's hosted API |

---

## Tech stack

- **FastAPI** — API framework
- **FAISS** — vector similarity search
- **Groq** (Llama 3.3 70B) — LLM inference
- **sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings with HF API fallback
- **Render** — deployment (free tier)

---

## Local setup

### Prerequisites
- Python 3.11+
- Groq API key — [console.groq.com](https://console.groq.com) (free)
- HuggingFace token (optional) — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Install

```bash
pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
# Add your GROQ_API_KEY (and optionally HF_TOKEN) to .env
```

### Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 for the chat UI, or use the API directly:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a senior Java developer"}
    ]
  }'
```

### Rebuild the catalog index

The normalized `data/catalog.json` is committed, so this is only needed if you update the raw source data:

```bash
python scripts/build_catalog.py
```

### Run the evaluation harness

Replays 10 conversation scenarios and measures Recall@10 + schema compliance:

```bash
python scripts/eval.py --url http://localhost:8000
# or against the live instance:
python scripts/eval.py --url https://shl-recommender-wtwd.onrender.com
```

---

## API reference

### `GET /health`

Returns `{"status": "ok"}` once the catalog index is ready. Returns `503` during startup.

### `POST /chat`

**Request**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a Java developer"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years experience"}
  ]
}
```

**Response**
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when the agent is clarifying, refusing, or comparing
- `end_of_conversation` is `true` when the user confirms the final shortlist
- `test_type` codes: `A` Ability, `B` Biodata/SJT, `C` Competencies, `D` Development, `K` Knowledge, `P` Personality, `S` Simulations, `E` Exercises

---

## Deployment

The repo includes `render.yaml` — connect it to Render and set two env vars:

| Variable | Value |
|---|---|
| `GROQ_API_KEY` | From [console.groq.com](https://console.groq.com) |
| `HF_TOKEN` | From [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

Render auto-deploys on every push to `main`. Cold starts rebuild the embedding cache (~2–3 min); subsequent restarts load from disk and are fast.
