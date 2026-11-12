
from pathlib import Path
import json
import uuid
import copy

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output" / "stable_chunks.json"

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


# Load original chunks
with INPUT_FILE.open("r", encoding="utf-8") as file:
    original_chunks = json.load(file)

# Create a separate copy for the edit test
edited_chunks = copy.deepcopy(original_chunks)

print("\nSTABLE ID EDIT TEST")
print("-" * 40)

# Simulate changing one sentence in company_policy.txt
old_sentence = "Passwords must contain at least 12 characters."
new_sentence = "Passwords must contain at least 14 characters."

for method, chunks in edited_chunks.items():
    for chunk in chunks:
        if chunk["source"] == "company_policy.txt":
            if old_sentence in chunk["text"]:
                chunk["text"] = chunk["text"].replace(
                    old_sentence, new_sentence, 1
                )
                chunk["chunk_id"] = generate_chunk_id(
                    chunk, method
                )
                print(f"{method}: paragraph edited")
                break

# Compare original and edited IDs
changed_count = 0
unchanged_count = 0

for method in original_chunks:
    original_by_key = {
        (chunk["source"], chunk["page"], chunk["chunk_index"]):
        chunk["chunk_id"]
        for chunk in original_chunks[method]
    }

    edited_by_key = {
        (chunk["source"], chunk["page"], chunk["chunk_index"]):
        chunk["chunk_id"]
        for chunk in edited_chunks[method]
    }

    for key, original_id in original_by_key.items():
        edited_id = edited_by_key.get(key)

        if edited_id != original_id:
            changed_count += 1
        else:
            unchanged_count += 1

print("\nRESULT")
print("-" * 40)
print("Changed chunk IDs:", changed_count)
print("Unchanged chunk IDs:", unchanged_count)
print("Original documents were not modified.")
