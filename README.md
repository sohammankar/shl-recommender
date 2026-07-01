# SHL Assessment Recommender

Conversational agent for selecting SHL assessments, built for the SHL Labs AI Intern take-home assignment.

## Architecture at a glance

```
POST /chat (stateless)
  │
  ├─ 1. Build retrieval query from conversation history (last 3 user turns)
  ├─ 2. Hybrid search: FAISS semantic (70%) + keyword BM25 (30%) → top 15 candidates
  ├─ 3. Inject candidates as grounded context into LLM system prompt
  ├─ 4. Call Groq (Llama 3.3 70B) with conversation history
  ├─ 5. Parse JSON response
  └─ 6. Validate every recommended URL against catalog → return
```

**Key design decisions:**
- **Local embeddings** (`all-MiniLM-L6-v2`) — no paid API, no rate limits, 377 items fits in RAM
- **FAISS flat index** — exact search, correct for <1000 items (no approximation needed)
- **Hybrid retrieval** — semantic for paraphrased queries, keyword for exact product names
- **Single LLM call per turn** — stays within 30s timeout, easier to debug than multi-step pipelines
- **URL validation layer** — every recommendation is cross-checked against the catalog before returning

## Setup

### 1. Prerequisites
- Python 3.11+
- A free Groq API key from https://console.groq.com

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the catalog index
```bash
python scripts/build_catalog.py
```
This normalizes `data/shl_product_catalog.json` into `data/catalog.json`.
The FAISS index itself is built in memory at server startup.

### 4. Set environment variables
```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### 5. Run locally
```bash
uvicorn app.main:app --reload --port 8000
```

### 6. Test it
```bash
# Health check
curl http://localhost:8000/health

# Single chat turn
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"}
    ]
  }'

# Run the full evaluation harness against your local server
python scripts/eval.py --url http://localhost:8000
```

## Deployment (Render)

1. Push this repo to GitHub (include `data/shl_product_catalog.json`)
2. Go to https://render.com → New → Web Service → connect your GitHub repo
3. Render will auto-detect `render.yaml`
4. In the Render dashboard → Environment → add `GROQ_API_KEY` = your key
5. Deploy. First startup takes ~60-90s (model download). Subsequent restarts are faster.
6. Your endpoint will be: `https://shl-recommender.onrender.com`

## API Reference

### GET /health
Returns `{"status": "ok"}` with HTTP 200 when ready.

### POST /chat
**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a Java developer"},
    {"role": "assistant", "content": "What is their seniority level?"},
    {"role": "user", "content": "Mid-level, around 4 years"}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 4 assessments for a mid-level Java developer...",
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

`recommendations` is `[]` when the agent is clarifying or refusing.
`end_of_conversation` is `true` only when the user confirms the final shortlist.
