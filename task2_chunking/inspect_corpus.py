import json
import tiktoken
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR/ "output"/ "extracted_documents.json"

tokenizer = tiktoken.get_encoding("cl100k_base")

with INPUT_FILE.open("r", encoding="utf-8")as file:
    documents= json.load(file)

print("EXPERIMENT CORPUS TOKEN COUNTS")
print("-" * 45)

for document in documents:
    if document["source"] == "experiment_corpus.pdf":
        token_count = len(tokenizer.encode(document["text"]))

        print(
            f"Page {document['page']:2}: "
            f"{token_count:4} tokens"
        )
