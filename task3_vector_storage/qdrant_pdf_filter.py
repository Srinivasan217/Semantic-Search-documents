
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "output" / "qdrant_db"

COLLECTION_NAME = "document_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

client = QdrantClient(path=str(DB_PATH))

model = SentenceTransformer(MODEL_NAME)

query = "What is the password reset policy?"

query_vector = model.encode(
    query,
    normalize_embeddings=True
).tolist()

# Filter only PDF documents
pdf_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="doc_type",
            match=models.MatchValue(value="pdf")
        )
    ]
)

# Count all PDF chunks
pdf_count = client.count(
    collection_name=COLLECTION_NAME,
    count_filter=pdf_filter,
    exact=True
).count

# Search top 5 PDF chunks
results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    query_filter=pdf_filter,
    limit=5,
    with_payload=True
).points

print("\nPDF FILTERED SEARCH")
print("-" * 40)
print("Filter: doc_type = pdf")
print("Total PDF chunks:", pdf_count)
print("Search results:", len(results))

for number, result in enumerate(results, start=1):
    print(f"\nResult {number}")
    print("Source:", result.payload["source"])
    print("Page:", result.payload["page"])
    print("Document type:", result.payload["doc_type"])
    print("Score:", round(result.score, 4))
    print("Text:", result.payload["text"][:150])

client.close()
