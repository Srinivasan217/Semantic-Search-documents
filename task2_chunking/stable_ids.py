
from pathlib import Path
import json
import uuid

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "chunking_results.json"
OUTPUT_FILE = BASE_DIR / "output" / "stable_chunks.json"

# Fixed namespace makes UUID5 IDs reproducible
NAMESPACE = uuid.NAMESPACE_URL


def generate_chunk_id(chunk, method):
    unique_value = (
        f"{method}|"
        f"{chunk['source']}|"
        f"{chunk['page']}|"
        f"{chunk['chunk_index']}|"
        f"{chunk['text']}"
    )

    return str(uuid.uuid5(NAMESPACE, unique_value))


# Load existing chunks
with INPUT_FILE.open("r", encoding="utf-8") as file:
    results = json.load(file)


stable_results = {}

for method, chunks in results.items():
    stable_results[method] = []

    for chunk in chunks:
        chunk_copy = chunk.copy()

        chunk_copy["chunk_id"] = generate_chunk_id(
            chunk_copy, method
        )

        stable_results[method].append(chunk_copy)


# Save stable IDs
with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(stable_results, file, indent=4, ensure_ascii=False)


# Test: Generate IDs again and compare
print("\nSTABLE ID TEST")
print("-" * 40)

all_ids_stable = True

for method, chunks in results.items():
    for chunk in chunks:
        first_id = generate_chunk_id(chunk, method)
        second_id = generate_chunk_id(chunk, method)

        if first_id != second_id:
            all_ids_stable = False

print("Repeated indexing produces identical IDs:", all_ids_stable)

# Display examples
for method, chunks in stable_results.items():
    if chunks:
        print(f"\nMethod: {method}")
        print("Source:", chunks[0]["source"])
        print("Chunk ID:", chunks[0]["chunk_id"])

print(f"\nStable chunks saved to: {OUTPUT_FILE}")
