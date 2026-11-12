import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "task3_vector_storage" / "output" / "qdrant_db"

QUESTIONS_PATH = BASE_DIR / "task4_api" / "evaluation_questions.json"

OUTPUT_PATH = BASE_DIR / "task4_api" / "evaluation_results.json"

COLLECTION_NAME = "document_chunks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


# --------------------------------------------------
# LOAD QUESTIONS
# --------------------------------------------------

with QUESTIONS_PATH.open("r", encoding="utf-8") as file:
    questions = json.load(file)

print(f"Loaded {len(questions)} evaluation questions.")


# --------------------------------------------------
# LOAD MODEL AND DATABASE
# --------------------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Connecting to Qdrant...")

client = QdrantClient(path=str(DB_PATH))


# --------------------------------------------------
# SEARCH FUNCTION
# --------------------------------------------------

def perform_search(question, use_filter=False):

    query_vector = model.encode(
        question["question"],
        normalize_embeddings=True,
    ).tolist()

    query_filter = None

    if use_filter:

        doc_type = Path(
            question["expected_source"]
        ).suffix.lstrip(".").lower()

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(value=doc_type),
                )
            ]
        )

    start_time = time.perf_counter()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        query_filter=query_filter,
        with_payload=True,
    ).points

    elapsed = time.perf_counter() - start_time

    return results, elapsed


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def calculate_metrics(evaluation_rows):

    total = len(evaluation_rows)

    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks = []

    for row in evaluation_rows:

        ranks = row["matching_ranks"]

        if any(rank <= 1 for rank in ranks):
            hit_at_1 += 1

        if any(rank <= 3 for rank in ranks):
            hit_at_3 += 1

        if any(rank <= 5 for rank in ranks):
            hit_at_5 += 1

        reciprocal_rank = (
            1 / min(ranks) if ranks else 0
        )

        reciprocal_ranks.append(reciprocal_rank)

    return {
        "Hit@1": round(hit_at_1 / total, 4),
        "Hit@3": round(hit_at_3 / total, 4),
        "Hit@5": round(hit_at_5 / total, 4),
        "MRR": round(sum(reciprocal_ranks) / total, 4),
        "average_search_time_seconds": round(
            sum(row["search_time_seconds"] for row in evaluation_rows)
            / total,
            4,
        ),
    }


# --------------------------------------------------
# RUN EVALUATION
# --------------------------------------------------

def evaluate_configuration(use_filter=False):

    evaluation_rows = []

    configuration = (
        "Filtered Search" if use_filter
        else "Baseline Search"
    )

    print(f"\nRunning: {configuration}")

    for index, question in enumerate(questions, start=1):

        results, elapsed = perform_search(
            question,
            use_filter=use_filter,
        )

        matching_ranks = []

        for rank, result in enumerate(results, start=1):

            payload = result.payload or {}

            if (
                payload.get("source") == question["expected_source"]
                and payload.get("page") == question["expected_page"]
            ):
                matching_ranks.append(rank)

        evaluation_rows.append({
            "question": question["question"],
            "expected_source": question["expected_source"],
            "expected_page": question["expected_page"],
            "matching_ranks": matching_ranks,
            "search_time_seconds": round(elapsed, 6),
        })

        print(
            f"Q{index:02d}: "
            f"{'PASS' if matching_ranks else 'MISS'}"
            f" | Matching ranks: {matching_ranks}"
        )

    return {
        "configuration": configuration,
        "metrics": calculate_metrics(evaluation_rows),
        "details": evaluation_rows,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

try:

    baseline = evaluate_configuration(use_filter=False)

    filtered = evaluate_configuration(use_filter=True)

    final_results = {
        "total_questions": len(questions),
        "top_k": TOP_K,
        "configurations": [
            baseline,
            filtered,
        ],
    }

    print("\n" + "=" * 60)
    print("SEARCH EVALUATION COMPARISON")
    print("=" * 60)

    for result in final_results["configurations"]:

        print(f"\n{result['configuration']}")

        for metric, value in result["metrics"].items():
            print(f"{metric}: {value}")

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(final_results, file, indent=4)

    print(f"\nResults saved to: {OUTPUT_PATH}")

finally:

    client.close()
    print("\nQdrant connection closed.")
