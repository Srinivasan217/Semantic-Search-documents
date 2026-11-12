from pathlib import Path
from qdrant_client import QdrantClient
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR /"output"/ "qdrant_db"

client = QdrantClient(path=str(DB_PATH))

info = client.get_collection("document_chunks")

print("\nQDRANT INDEX STATUS")
print("-" * 40)
print("Collection points:", info.points_count)
print("Indexed vectors:", info.indexed_vectors_count)
print("Status:", info.status)

print("\nHNSW CONFIGURATION")
print("-" * 40)

print("M:", info.config.hnsw_config.m)
print("EF Construct:", info.config.hnsw_config.ef_construct)

client.close()

