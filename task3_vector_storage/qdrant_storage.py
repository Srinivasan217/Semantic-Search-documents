
from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent

INPUT_FILE = (
    PROJECT_DIR
    / "task2_chunking"
    / "output"
    / "stable_chunks.json"
)

DB_PATH = BASE_DIR / "output" / "qdrant_db"
COLLECTION_NAME = "document_chunks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384


# 1. Load sentence-based chunks
with INPUT_FILE.open("r", encoding="utf-8") as file:
    all_chunks = json.load(file)

chunks = all_chunks["sentence_based"]

print("Chunks loaded:", len(chunks))


# 2. Load embedding model
print("\nLoading embedding model...")
model = SentenceTransformer(MODEL_NAME)

texts = [chunk["text"] for chunk in chunks]

vectors = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Embedding shape:", vectors.shape)


# 3. Create local Qdrant database
client = QdrantClient(path=str(DB_PATH))

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE
        ),
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100
        )
    )

    print("\nCollection created.")
else:
    print("\nCollection already exists.")


# 4. Create metadata indexes
for field, schema in [
    ("source", models.PayloadSchemaType.KEYWORD),
    ("doc_type", models.PayloadSchemaType.KEYWORD),
    ("page", models.PayloadSchemaType.INTEGER),
]:
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=field,
        field_schema=schema
    )


# 5. Prepare points
points = []

for chunk, vector in zip(chunks, vectors):
    payload = {
        "text": chunk["text"],
        "source": chunk["source"],
        "page": chunk["page"],
        "section_heading": chunk["section_heading"],
        "created_date": chunk["created_date"],
        "chunk_index": chunk["chunk_index"],
        "doc_type": chunk["doc_type"],
        "method": "sentence_based"
    }

    points.append(
        models.PointStruct(
            id=chunk["chunk_id"],
            vector=vector.tolist(),
            payload=payload
        )
    )


# 6. Upsert points
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True
)

first_count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

print("\nVECTOR STORAGE")
print("-" * 40)
print("Points stored:", first_count)


# 7. Re-upsert to test duplicate prevention
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
    wait=True
)

second_count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

print("\nRE-UPsert TEST")
print("-" * 40)
print("Count after first upsert:", first_count)
print("Count after second upsert:", second_count)
print("Duplicate prevention:", first_count == second_count)


# 8. Normal semantic search
query = "How often must employees reset their passwords?"

query_vector = model.encode(
    query,
    normalize_embeddings=True
).tolist()

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=5,
    with_payload=True
).points

print("\nSEMANTIC SEARCH")
print("-" * 40)

for result in results:
    print("Source:", result.payload["source"])
    print("Page:", result.payload["page"])
    print("Score:", round(result.score, 4))
    print("Text:", result.payload["text"][:150])
    print("-" * 30)


# 9. Filter by document type
doc_type = chunks[0]["doc_type"]

doc_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="doc_type",
            match=models.MatchValue(value=doc_type)
        )
    ]
)

doc_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=doc_filter,
    limit=5,
    with_payload=True
).points

print("\nDOCUMENT TYPE FILTER")
print("-" * 40)
print("Filter:", doc_type)
print("Matching results:", len(doc_results))


# 10. Filter pages greater than or equal to 3
page_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="page",
            range=models.Range(gte=3)
        )
    ]
)

page_results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=page_filter,
    limit=5,
    with_payload=True
).points

print("\nPAGE FILTER")
print("-" * 40)
print("Required page: >= 3")
print("Matching results:", len(page_results))

for result in page_results:
    print(result.payload["source"], "Page:", result.payload["page"])


# 11. Delete one source and verify
delete_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="source",
            match=models.MatchValue(value="company_policy.txt")
        )
    ]
)

before_delete = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=models.FilterSelector(
        filter=delete_filter
    ),
    wait=True
)

after_delete = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

print("\nDELETE SOURCE TEST")
print("-" * 40)
print("Before deletion:", before_delete)
print("After deletion:", after_delete)


# 12. Restore deleted points for future tasks
restore_points = [
    point for point in points
    if point.payload["source"] == "company_policy.txt"
]

client.upsert(
    collection_name=COLLECTION_NAME,
    points=restore_points,
    wait=True
)

final_count = client.count(
    collection_name=COLLECTION_NAME,
    exact=True
).count

print("\nFINAL STORAGE CHECK")
print("-" * 40)
print("Points after restoration:", final_count)
print("Original count restored:", final_count == first_count)

client.close()

print("\nQdrant experiment completed.")
