from pathlib import Path
import time

import faiss
import numpy as np

#Configuration
NUM_VECTORS = 50_000
DIMENSIONS = 384
NUM_QUERIES = 100
TOP_K = 10
EF_SEARCH_VALUES = [64, 128, 256, 512]

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate reproducible random vectors
np.random.seed(42)


vectors = np.random.random(
    (NUM_VECTORS, DIMENSIONS)
).astype("float32")

queries = np.random.random(
    (NUM_QUERIES, DIMENSIONS)
).astype("float32")

# Normalize vectors for cosine similarity
faiss.normalize_L2(vectors)
faiss.normalize_L2(queries)


# 1. Exact search using IndexFlatIP
print("\nEXACT SEARCH")
print("-" * 40)

exact_index = faiss.IndexFlatIP(DIMENSIONS)

start_time = time.perf_counter()
exact_index.add(vectors)
exact_build_time = time.perf_counter() - start_time

start_time = time.perf_counter()
exact_scores, exact_results = exact_index.search(
    queries, TOP_K
)
exact_search_time = time.perf_counter() - start_time

print("Vectors indexed:", exact_index.ntotal)
print(f"Build time: {exact_build_time:.4f} seconds")
print(f"Search time: {exact_search_time:.4f} seconds")


# 2. Approximate search using HNSW
print("\nHNSW APPROXIMATE SEARCH")
print("-" * 40)

hnsw_index = faiss.IndexHNSWFlat(
    DIMENSIONS, 32, faiss.METRIC_INNER_PRODUCT
)

hnsw_index.hnsw.efConstruction = 100


start_time = time.perf_counter()
hnsw_index.add(vectors)
hnsw_build_time = time.perf_counter() - start_time


print(f"HNSW build time: {hnsw_build_time:.4f} seconds")

# Test different efSearch values
results = []

print("\nHNSW TUNING RESULTS")
print("-" * 60)
print(f"{'efSearch':<12}{'Time (s)':<15}{'Recall@10':<15}")

for ef_search in EF_SEARCH_VALUES:

    hnsw_index.hnsw.efSearch = ef_search

    start = time.perf_counter()
    hnsw_scores, hnsw_results = hnsw_index.search(
        queries, TOP_K
    )
    search_time = time.perf_counter() - start

# 3. Calculate Recall@10
recall_values = []

for exact, approximate in zip(exact_results, hnsw_results):
    exact_set = set(exact)
    approximate_set = set(approximate)

    recall = (len(exact_set.intersection(approximate_set)) / TOP_K)
    recall_values.append(recall)

average_recall = float(np.mean(recall_values))
results.append({
        "efSearch": ef_search,
        "search_time": search_time,
        "recall": average_recall
    })
print(
        f"{ef_search:<12}"
        f"{search_time:<15.4f}"
        f"{average_recall * 100:.2f}%"
    )


# Best recall configuration
best_result = max(results, key=lambda item: item["recall"])

print("\nBEST RECALL CONFIGURATION")
print("-" * 40)
print("efSearch:", best_result["efSearch"])
print(f"Recall@10: {best_result['recall'] * 100:.2f}%")

# Save report
output_file = OUTPUT_DIR / "faiss_benchmark.txt"

with output_file.open("w", encoding="utf-8") as file:
    file.write(f"Vectors: {NUM_VECTORS}\n")
    file.write(f"Dimensions: {DIMENSIONS}\n")
    file.write(f"Queries: {NUM_QUERIES}\n")
    file.write(f"Exact build time: {exact_build_time:.4f}s\n")
    file.write(f"Exact search time: {exact_search_time:.4f}s\n")
    file.write(f"HNSW build time: {hnsw_build_time:.4f}s\n\n")

    for result in results:
        file.write(
            f"efSearch={result['efSearch']}, "
            f"time={result['search_time']:.4f}s, "
            f"recall={result['recall'] * 100:.2f}%\n"
        )

print("\nResults saved to:", output_file)
