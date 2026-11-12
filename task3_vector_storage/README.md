# Task 3 — Vector Storage and ANN Search

## 1. ANN and HNSW Concepts

- ANN means Approximate Nearest Neighbor search.
- It finds vectors close to a query vector without checking every vector.
- HNSW means Hierarchical Navigable Small World.
- HNSW uses a graph structure to find similar vectors efficiently.
- It trades some search accuracy for faster retrieval.

## 2. FAISS Benchmark

Experiment configuration:
- Number of vectors: 50,000
- Vector dimensions: 384
- Number of queries: 100
- Top K: 10

### Exact Search vs HNSW

| Measurement | Exact Search | HNSW |
|---|---:|---:|
| Build time | 0.1160 seconds | 54.6897 seconds |
| Search time | 0.8906 seconds | 0.3952 seconds |
| Recall@10 | 100% | 92.00% |

Best HNSW configuration:
- efSearch: 512
- Recall@10: 92.00%

Increasing efSearch improved recall from the initial 52.90% to 92.00%.

Note: Benchmark timings depend on hardware, configuration, and system load.

## 3. Qdrant Configuration

- Database: Local Qdrant
- Collection: `document_chunks`
- Embedding model: `all-MiniLM-L6-v2`
- Vector dimensions: 384
- Distance: Cosine
- HNSW M: 16
- EF Construct: 100

## 4. Qdrant Test Results

| Test | Result |
|---|---|
| Stored chunks | 35 |
| Re-upsert | Count remained 35 |
| Semantic search | Successful |
| Document-type filter (`txt`) | 3 results |
| Page >= 3 filter | 5 results |
| PDF-only filter | 5 results |
| Total PDF chunks | 31 |
| Delete source | Count changed from 35 to 34 |
| Restore source | Count returned to 35 |
| Collection status | green |

PDF filtering was tested using `doc_type = pdf`. The search successfully returned five PDF chunks.

Note: Local Qdrant reported zero indexed vectors. Local embedded mode does not expose server-style payload indexing and HNSW indexing statistics.

## 5. Conclusion

FAISS demonstrated the speed-versus-accuracy trade-off between exact search and HNSW approximate search.

After tuning efSearch to 512, HNSW achieved 92.00% Recall@10.

Qdrant successfully stored document embeddings, metadata, and stable chunk IDs. Semantic search, metadata filtering, PDF filtering, duplicate prevention, deletion, and restoration were tested successfully.
