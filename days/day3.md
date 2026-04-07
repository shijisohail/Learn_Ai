# Day 3 — Memory & RAG

> Memory is the first architectural decision you make when building a real agent. Get it wrong and you'll hit context limits at scale, lose important information, and pay through the nose for tokens. Get it right and your agents become genuinely useful across long sessions.

---

## 1. The Memory Problem

An agent has four types of memory, each with different trade-offs:

| Type | Storage | Access | Latency | Capacity | Best For |
|---|---|---|---|---|---|
| **In-context** | Messages array | Direct (model reads it) | 0ms | ~200K tokens | Active session, current task |
| **Episodic (Vector)** | Vector DB | Semantic similarity search | 10-50ms | Unlimited | Past interactions, facts, docs |
| **Semantic (KB)** | Vector DB + structured | Hybrid search | 20-100ms | Unlimited | Domain knowledge, reference data |
| **Procedural** | System prompt | Always present | 0ms | ~10K tokens | Rules, persona, capabilities |

### Decision Framework

```
Question: "What does the agent need to remember?"

Is it needed EVERY turn?
  → Procedural: put in system prompt

Is it from THIS session, likely relevant to next few turns?
  → In-context: keep in messages array

Is it from PAST sessions or a large corpus?
  → Episodic/Semantic: vector database

Is it a fixed rule or behavioral constraint?
  → Procedural: system prompt
```

**The Fundamental Tension:** In-context memory is free (already loaded) but finite. External memory is unlimited but requires retrieval that may miss relevant content.

---

## 2. Embeddings Deep Dive

### What an Embedding Is

An embedding is a dense floating-point vector that represents the semantic meaning of a piece of text. Produced by an encoder model (not a generative model), the vector captures "meaning" in a way that allows mathematical comparison.

```python
from anthropic import Anthropic
# Anthropic doesn't provide embeddings; use OpenAI or Voyage
import voyageai

vo = voyageai.Client()

# Two semantically similar sentences
result = vo.embed(
    ["connection pool exhausted", "too many database connections"],
    model="voyage-3"
)

v1, v2 = result.embeddings[0], result.embeddings[1]
print(f"Vector dimension: {len(v1)}")  # 1024 for voyage-3
```

### Cosine Similarity

Similarity is measured as the angle between vectors, normalized to [-1, 1]:

```python
import numpy as np

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Example similarities (approximate)
# "cat" vs "kitten"     → ~0.92
# "cat" vs "dog"        → ~0.78
# "cat" vs "database"   → ~0.31
# "fastapi" vs "flask"  → ~0.85
```

### Embedding Model Selection

| Model | Dimensions | Cost | Use Case |
|---|---|---|---|
| `voyage-3` | 1024 | $0.06/MTok | Production, highest quality |
| `voyage-3-lite` | 512 | $0.02/MTok | Cost-sensitive, good quality |
| `text-embedding-3-small` | 1536 | $0.02/MTok | OpenAI ecosystem |
| `text-embedding-3-large` | 3072 | $0.13/MTok | Highest OpenAI quality |

> Use voyage-3 for production RAG. It consistently outperforms OpenAI embeddings on retrieval benchmarks, especially for technical/code content.

### Embedding Batching

```python
import voyageai
from typing import Generator

def batch_embed(texts: list[str], model: str = "voyage-3", batch_size: int = 128) -> list[list[float]]:
    """Embed a large list of texts in batches."""
    vo = voyageai.Client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = vo.embed(batch, model=model, input_type="document")
        all_embeddings.extend(result.embeddings)

    return all_embeddings
```

---

## 3. Vector Databases

### How They Work

Vector databases store vectors alongside metadata and support Approximate Nearest Neighbor (ANN) search — finding the K vectors most similar to a query vector without scanning every stored vector. This makes sub-100ms retrieval possible with millions of vectors.

**Indexing Algorithms:**

**HNSW (Hierarchical Navigable Small World)** — Graph-based. Build time is slow (O(n log n)), but query time is very fast (O(log n)). Most common for production. High memory usage.

**IVF (Inverted File Index)** — Cluster vectors into N lists (Voronoi cells), search only the nearest M clusters. Faster to build, lower memory, slightly lower accuracy.

**IVF-PQ (Product Quantization)** — Compresses vectors to reduce memory at cost of some accuracy. Use when memory is constrained.

### pgvector (Your Best Starting Point)

You already know PostgreSQL. pgvector adds a `vector` type and ANN indexes. For <1M documents, this is often sufficient and removes a separate service dependency.

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE documents (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(1024),          -- voyage-3 dimension
    metadata    JSONB DEFAULT '{}',
    source      VARCHAR(500),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index (recommended for query performance)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Or IVF if you need less memory
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Similarity search query
SELECT
    id,
    content,
    metadata,
    1 - (embedding <=> $1::vector) AS similarity
FROM documents
WHERE metadata->>'source' = 'internal_docs'   -- metadata filter
ORDER BY embedding <=> $1::vector              -- <=> is cosine distance
LIMIT 10;
```

### Vector Database Comparison

| Database | Deployment | Scaling | Best For |
|---|---|---|---|
| **pgvector** | Your existing Postgres | Up to ~5M vectors | Existing Postgres shops, <5M docs |
| **Qdrant** | Self-host / Cloud | 100M+ vectors | High performance, complex filters |
| **Pinecone** | Managed cloud | Unlimited | No-ops, serverless |
| **Chroma** | Embedded / Self-host | Small-medium | Development, prototyping |
| **Weaviate** | Self-host / Cloud | 100M+ vectors | Multi-tenant, hybrid search |

---

## 4. The RAG Pipeline

### Full Architecture

```
Documents
    ↓ Load
Raw Text
    ↓ Chunk
Text Chunks (with metadata)
    ↓ Embed
Vectors
    ↓ Store
Vector Database
    ↑ Query Vector (embed user question)
Retrieved Chunks
    ↓ Rerank (optional)
Top-K Chunks
    ↓ Inject into prompt
LLM Response
```

### Complete Implementation

```python
import asyncio
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import psycopg2
from psycopg2.extras import execute_values
import voyageai
import anthropic

@dataclass
class Document:
    content: str
    source: str
    metadata: dict = field(default_factory=dict)

@dataclass
class Chunk:
    content: str
    source: str
    metadata: dict
    chunk_index: int
    embedding: Optional[list[float]] = None

    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()


class RAGPipeline:
    def __init__(self, db_url: str, voyage_api_key: str, embedding_model: str = "voyage-3"):
        self.db_url = db_url
        self.vo = voyageai.Client(api_key=voyage_api_key)
        self.embedding_model = embedding_model
        self.embedding_dim = 1024  # voyage-3

    def get_connection(self):
        return psycopg2.connect(self.db_url)

    def setup_schema(self):
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id          BIGSERIAL PRIMARY KEY,
                        content     TEXT NOT NULL,
                        content_hash VARCHAR(32) UNIQUE,
                        source      VARCHAR(500),
                        chunk_index INTEGER,
                        metadata    JSONB DEFAULT '{}',
                        embedding   vector(%s),
                        created_at  TIMESTAMPTZ DEFAULT NOW()
                    )
                """, (self.embedding_dim,))
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
                    ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
            conn.commit()

    # ---- INGESTION PIPELINE ----

    def chunk_document(
        self,
        doc: Document,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[Chunk]:
        """Recursive character-based chunking with overlap."""
        words = doc.content.split()
        chunks = []
        i = 0

        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)

            chunks.append(Chunk(
                content=chunk_content,
                source=doc.source,
                metadata=doc.metadata,
                chunk_index=len(chunks),
            ))

            # Advance by chunk_size - overlap
            i += chunk_size - overlap
            if i >= len(words):
                break

        return chunks

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed chunks in batches of 128."""
        BATCH_SIZE = 128
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [c.content for c in batch]
            result = self.vo.embed(texts, model=self.embedding_model, input_type="document")
            for chunk, embedding in zip(batch, result.embeddings):
                chunk.embedding = embedding
        return chunks

    def store_chunks(self, chunks: list[Chunk]) -> int:
        """Upsert chunks into the database. Returns count inserted."""
        rows = [
            (c.content, c.content_hash, c.source, c.chunk_index,
             c.metadata, c.embedding)
            for c in chunks if c.embedding is not None
        ]

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO rag_chunks
                        (content, content_hash, source, chunk_index, metadata, embedding)
                    VALUES %s
                    ON CONFLICT (content_hash) DO NOTHING
                """, rows, template="(%s, %s, %s, %s, %s, %s::vector)")
            conn.commit()
            return cur.rowcount

    def ingest(self, documents: list[Document], chunk_size: int = 512) -> dict:
        """Full ingestion pipeline."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc, chunk_size=chunk_size)
            all_chunks.extend(chunks)

        all_chunks = self.embed_chunks(all_chunks)
        inserted = self.store_chunks(all_chunks)

        return {
            "documents": len(documents),
            "chunks_created": len(all_chunks),
            "chunks_inserted": inserted,
        }

    # ---- RETRIEVAL PIPELINE ----

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[dict] = None,
    ) -> list[dict]:
        """Retrieve top-k chunks similar to query."""
        # Embed the query
        result = self.vo.embed([query], model=self.embedding_model, input_type="query")
        query_vector = result.embeddings[0]

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                if metadata_filter:
                    cur.execute("""
                        SELECT id, content, source, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM rag_chunks
                        WHERE metadata @> %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_vector, metadata_filter, query_vector, top_k))
                else:
                    cur.execute("""
                        SELECT id, content, source, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM rag_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_vector, query_vector, top_k))

                rows = cur.fetchall()

        return [
            {
                "id": row[0],
                "content": row[1],
                "source": row[2],
                "metadata": row[3],
                "similarity": float(row[4]),
            }
            for row in rows
        ]

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        model: str = "claude-sonnet-4-5",
    ) -> str:
        """Generate answer using retrieved context."""
        context = "\n\n---\n\n".join([
            f"Source: {chunk['source']} (similarity: {chunk['similarity']:.3f})\n{chunk['content']}"
            for chunk in retrieved_chunks
        ])

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=(
                "You are a technical assistant. Answer questions based ONLY on the provided context. "
                "If the context doesn't contain enough information, say so clearly. "
                "Always cite the source when referencing specific information."
            ),
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }]
        )
        return response.content[0].text

    def query(self, question: str, top_k: int = 5) -> dict:
        """Full RAG query: retrieve + generate."""
        chunks = self.retrieve(question, top_k=top_k)
        answer = self.generate(question, chunks)
        return {
            "answer": answer,
            "sources": [c["source"] for c in chunks],
            "chunks_used": len(chunks),
            "top_similarity": chunks[0]["similarity"] if chunks else 0,
        }
```

---

## 5. Chunking Strategies

### Fixed-Size with Overlap (Default)

Split by word count with a sliding window overlap:

```python
def chunk_fixed_size(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks
```

**When to use:** Documents without clear structure (logs, raw text). Simple and predictable.

### Semantic Chunking

Split at natural semantic boundaries (sentences, paragraphs):

```python
import re

def chunk_by_paragraph(text: str, min_size: int = 100, max_size: int = 1000) -> list[str]:
    """Split on double newlines, merge short paragraphs."""
    paragraphs = re.split(r'\n\n+', text.strip())
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) < max_size:
            current = (current + "\n\n" + para).strip()
        else:
            if len(current) >= min_size:
                chunks.append(current)
            current = para

    if current and len(current) >= min_size:
        chunks.append(current)

    return chunks
```

**When to use:** Markdown docs, articles, structured text. Preserves natural context boundaries.

### Recursive Character Splitting (LangChain standard)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],  # Try each in order
)

chunks = splitter.split_text(document_text)
```

### Chunking Strategy Comparison

| Strategy | Pros | Cons | Default? |
|---|---|---|---|
| Fixed-size | Predictable, simple | Breaks mid-sentence | Good for code |
| Paragraph | Semantic boundaries | Variable size | Good for docs |
| Sentence | Precise | Too small usually | Combine with merging |
| Recursive | Adaptive | Complex | LangChain default |

**Chunk size guidelines:**
- Too small (<100 tokens): No context, high retrieval noise
- Sweet spot (256-512 tokens): Good precision + context
- Too large (>1000 tokens): Low precision, wrong sections retrieved

---

## 6. Retrieval Strategies

### Basic Similarity Search

Covered in section 4. Top-K by cosine similarity. Simple, works well.

### Maximum Marginal Relevance (MMR)

Balances relevance with diversity — avoids returning 5 nearly-identical chunks:

```python
import numpy as np

def mmr_rerank(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.5,
) -> list[dict]:
    """
    MMR selection: balance relevance (similarity to query)
    with diversity (dissimilarity to already-selected docs).
    lambda=1.0 → pure relevance; lambda=0.0 → pure diversity
    """
    query = np.array(query_embedding)
    embeddings = np.array(candidate_embeddings)

    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            # Relevance to query
            relevance = np.dot(query, embeddings[idx]) / (
                np.linalg.norm(query) * np.linalg.norm(embeddings[idx]) + 1e-8
            )

            # Diversity from already selected
            if selected_indices:
                selected_embs = embeddings[selected_indices]
                redundancy = max(
                    np.dot(embeddings[idx], sel) / (
                        np.linalg.norm(embeddings[idx]) * np.linalg.norm(sel) + 1e-8
                    )
                    for sel in selected_embs
                )
            else:
                redundancy = 0.0

            score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if score > best_score:
                best_score = score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]
```

### Hybrid Search (BM25 + Semantic)

Combine keyword-based (BM25) with vector similarity for better coverage:

```sql
-- Requires: pg_search or manually computing BM25 scores
-- Simplified version using PostgreSQL full-text search + vector

SELECT
    id, content, source,
    ts_rank(to_tsvector('english', content), query) AS bm25_score,
    1 - (embedding <=> $1::vector) AS vector_score,
    -- Reciprocal Rank Fusion
    (1.0 / (60 + ts_rank(to_tsvector('english', content), query) * 100)) +
    (1.0 / (60 + (embedding <=> $1::vector) * 100)) AS rrf_score
FROM rag_chunks,
     plainto_tsquery('english', $2) query
WHERE to_tsvector('english', content) @@ query
   OR (embedding <=> $1::vector) < 0.3  -- union of results
ORDER BY rrf_score DESC
LIMIT 10;
```

### Re-ranking with a Cross-Encoder

After initial retrieval, re-rank with a more expensive but accurate model:

```python
import voyageai

def rerank_results(
    query: str,
    documents: list[str],
    top_k: int = 5,
) -> list[int]:
    """Use Voyage reranker to re-score initial retrieval results."""
    vo = voyageai.Client()
    result = vo.rerank(
        query=query,
        documents=documents,
        model="rerank-2",
        top_k=top_k,
    )
    return [r.index for r in result.results]
```

**Retrieval strategy selection:**

| Use Case | Strategy |
|---|---|
| Single domain, high precision needed | Semantic only |
| Mixed technical + natural language | Hybrid (BM25 + semantic) |
| Diverse document corpus | MMR for diversity |
| Critical applications (code review, legal) | Semantic → Rerank |

---

## 7. Conversation Memory

### Short-Term: Sliding Window

Keep the last N messages in context. Simple but loses early context:

```python
class SlidingWindowMemory:
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: list[dict] = []

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    @property
    def messages(self) -> list[dict]:
        return self._messages[-self.max_messages:]

    def clear(self):
        self._messages = []
```

### Long-Term: Summarization + Vector Retrieval

```python
import anthropic
import voyageai
from datetime import datetime

class EpisodicMemory:
    """
    Maintains both a rolling summary and searchable episode store.
    When conversation exceeds threshold, summarize and store in vector DB.
    """

    def __init__(self, rag_pipeline: RAGPipeline, session_id: str):
        self.rag = rag_pipeline
        self.session_id = session_id
        self.client = anthropic.Anthropic()
        self._recent: list[dict] = []
        self._summary: str = ""
        self.ARCHIVE_THRESHOLD = 10  # Archive after 10 turns

    def add(self, role: str, content: str):
        self._recent.append({"role": role, "content": content})
        if len(self._recent) >= self.ARCHIVE_THRESHOLD:
            self._archive_and_compress()

    def _archive_and_compress(self):
        """Summarize recent messages and store in vector DB."""
        # Create summary
        messages_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in self._recent
        ])

        response = self.client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize this conversation in 3-4 sentences, "
                    f"preserving key facts, decisions, and context:\n\n{messages_text}"
                )
            }]
        )
        new_summary = response.content[0].text

        # Accumulate summary
        self._summary = f"{self._summary}\n\n{new_summary}".strip()

        # Store in vector DB for future retrieval
        self.rag.ingest([
            Document(
                content=messages_text,
                source=f"session:{self.session_id}:{datetime.utcnow().isoformat()}",
                metadata={"session_id": self.session_id, "type": "episodic"}
            )
        ], chunk_size=256)

        # Keep only last 4 messages in recent
        self._recent = self._recent[-4:]

    def get_context(self, current_query: str) -> str:
        """Build context from summary + relevant past episodes."""
        context_parts = []

        if self._summary:
            context_parts.append(f"Session summary so far:\n{self._summary}")

        # Retrieve relevant past episodes
        relevant = self.rag.retrieve(
            current_query,
            top_k=3,
            metadata_filter={"session_id": self.session_id}
        )

        if relevant:
            episodes = "\n\n".join(c["content"][:500] for c in relevant)
            context_parts.append(f"Relevant past context:\n{episodes}")

        return "\n\n---\n\n".join(context_parts)

    @property
    def messages(self) -> list[dict]:
        return self._recent
```

---

## 8. Production Notes

### Embedding Cache

Avoid re-embedding the same text repeatedly:

```python
import hashlib
import json
from functools import lru_cache
import redis

class CachedEmbedder:
    def __init__(self, redis_url: str, model: str = "voyage-3", ttl: int = 86400 * 7):
        self.redis = redis.from_url(redis_url)
        self.vo = voyageai.Client()
        self.model = model
        self.ttl = ttl  # 7 days

    def embed(self, text: str) -> list[float]:
        cache_key = f"emb:{self.model}:{hashlib.md5(text.encode()).hexdigest()}"
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        result = self.vo.embed([text], model=self.model, input_type="document")
        embedding = result.embeddings[0]

        self.redis.setex(cache_key, self.ttl, json.dumps(embedding))
        return embedding
```

### Async Embedding Generation

For ingestion pipelines processing large document sets:

```python
import asyncio
import aiohttp

async def embed_batch_async(texts: list[str], api_key: str) -> list[list[float]]:
    """Call Voyage embeddings API asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": texts, "model": "voyage-3", "input_type": "document"}
        ) as resp:
            data = await resp.json()
            return [d["embedding"] for d in data["data"]]

async def ingest_async(documents: list[Document], batch_size: int = 128) -> None:
    """Async ingestion with concurrent embedding calls."""
    all_chunks = [
        chunk
        for doc in documents
        for chunk in chunk_document(doc)
    ]

    # Create batches
    batches = [all_chunks[i:i+batch_size] for i in range(0, len(all_chunks), batch_size)]

    # Embed all batches concurrently (respect rate limits)
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent embedding calls

    async def embed_with_semaphore(batch):
        async with semaphore:
            texts = [c.content for c in batch]
            embeddings = await embed_batch_async(texts, os.environ["VOYAGE_API_KEY"])
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb

    await asyncio.gather(*[embed_with_semaphore(b) for b in batches])
```

### Retrieval Evaluation (Recall@K)

```python
def evaluate_retrieval(
    rag: RAGPipeline,
    eval_dataset: list[dict],  # [{"query": str, "relevant_doc_ids": list[int]}]
    k: int = 5,
) -> dict:
    """
    Compute recall@k: what fraction of relevant docs are in the top-k results?
    """
    recalls = []

    for item in eval_dataset:
        results = rag.retrieve(item["query"], top_k=k)
        retrieved_ids = {r["id"] for r in results}
        relevant_ids = set(item["relevant_doc_ids"])

        if not relevant_ids:
            continue

        recall = len(retrieved_ids & relevant_ids) / len(relevant_ids)
        recalls.append(recall)

    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        "num_queries": len(recalls),
    }
```

---

## 9. Code Examples

### Full RAG with pgvector

See section 4 for the complete `RAGPipeline` class. Usage:

```python
import os

rag = RAGPipeline(
    db_url=os.environ["DATABASE_URL"],
    voyage_api_key=os.environ["VOYAGE_API_KEY"],
)
rag.setup_schema()

# Ingest
docs = [
    Document(
        content="FastAPI is a modern Python web framework...",
        source="fastapi-docs",
        metadata={"category": "framework", "language": "python"}
    ),
]
stats = rag.ingest(docs)
print(stats)  # {"documents": 1, "chunks_created": 3, "chunks_inserted": 3}

# Query
result = rag.query("How do I handle async database connections in FastAPI?")
print(result["answer"])
print(f"Sources: {result['sources']}")
```

### Hybrid Retrieval Function

```python
from dataclasses import dataclass

@dataclass
class HybridSearchResult:
    id: int
    content: str
    source: str
    vector_score: float
    text_score: float
    rrf_score: float

def hybrid_search(
    query: str,
    query_embedding: list[float],
    conn,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    """Reciprocal Rank Fusion of BM25 and vector search."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH vector_results AS (
                SELECT id, content, source,
                       row_number() OVER (ORDER BY embedding <=> %s::vector) AS vr_rank,
                       1 - (embedding <=> %s::vector) AS vector_score
                FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 50
            ),
            text_results AS (
                SELECT id, content, source,
                       row_number() OVER (ORDER BY ts_rank_cd(
                           to_tsvector('english', content),
                           plainto_tsquery('english', %s)
                       ) DESC) AS tr_rank,
                       ts_rank_cd(
                           to_tsvector('english', content),
                           plainto_tsquery('english', %s)
                       ) AS text_score
                FROM rag_chunks
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                ORDER BY text_score DESC
                LIMIT 50
            )
            SELECT
                COALESCE(v.id, t.id) AS id,
                COALESCE(v.content, t.content) AS content,
                COALESCE(v.source, t.source) AS source,
                COALESCE(v.vector_score, 0) AS vector_score,
                COALESCE(t.text_score, 0) AS text_score,
                (1.0 / (%s + COALESCE(v.vr_rank, 1000))) +
                (1.0 / (%s + COALESCE(t.tr_rank, 1000))) AS rrf_score
            FROM vector_results v
            FULL OUTER JOIN text_results t ON v.id = t.id
            ORDER BY rrf_score DESC
            LIMIT %s
        """, (query_embedding, query_embedding, query_embedding,
              query, query, query, rrf_k, rrf_k, top_k))

        return [
            HybridSearchResult(
                id=row[0], content=row[1], source=row[2],
                vector_score=row[3], text_score=row[4], rrf_score=row[5]
            )
            for row in cur.fetchall()
        ]
```

---

## 10. Key Concepts Cheatsheet

| Concept | One-liner |
|---|---|
| Embedding | Dense vector representing semantic meaning of text |
| Cosine similarity | Angle between vectors; 1.0 = identical meaning |
| HNSW | Graph-based ANN index; fast queries, high memory |
| IVF | Cluster-based ANN index; lower memory, slightly slower |
| pgvector | Postgres extension; `<=>` operator for cosine distance |
| Chunking | Split documents into ~512-token pieces before embedding |
| Overlap | Shared tokens between adjacent chunks preserve boundary context |
| RAG | Retrieve relevant chunks, inject into prompt, generate answer |
| MMR | Diversify retrieval results; balances relevance with variety |
| Hybrid search | BM25 + vector similarity; better recall than either alone |
| RRF | Reciprocal Rank Fusion; merge ranked lists from multiple retrievers |
| Recall@K | What fraction of relevant docs appear in top-K results |

---

## 11. Day 3 Exercises

**Exercise 1: Chunking Comparison**
Take a 2000-word technical article. Chunk it with (a) fixed-size 256 tokens, (b) fixed-size 512 tokens, (c) paragraph-based. Embed all chunks with voyage-3. Query "how does the performance scale" and compare which chunking strategy returns the most relevant top-3 results.
_Expected: Comparison table with similarity scores per strategy_

**Exercise 2: pgvector RAG**
Set up a local PostgreSQL with pgvector. Ingest 10 Python documentation pages. Query it with 5 different questions. Measure: (a) retrieval latency, (b) top similarity score, (c) whether the answer is correct. Log results to a CSV.
_Expected: Working RAG pipeline with latency measurements_

**Exercise 3: Conversation Memory**
Build the `EpisodicMemory` class from section 7. Run a 30-turn conversation where the user tells the agent details about their system architecture early on. In turn 25, ask about those early details. Verify the agent retrieves them correctly via vector search.
_Expected: Agent recalls details from >10 turns ago via episodic memory_

**Exercise 4: Hybrid Search**
Implement the hybrid search query from section 9. Compare results for a technical query ("PostgreSQL connection pool settings") against pure vector search. Count how many of the top-10 results differ between the two approaches.
_Expected: Hybrid search returns at least 2-3 different results vs. pure vector, including keyword-matched docs_
