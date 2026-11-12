import json
from pathlib import Path

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR/"output"

DOCUMENTS_FILE = OUTPUT_DIR / "extracted_documents.json"
QUESTION_FILE = OUTPUT_DIR / "evaluation_questions.json"

MODEL_NAME ="sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZES = [128, 256,512, 1024]
OVERLAP=50

tokenizer = tiktoken.get_encoding("cl100k_base")

#Load documents and questions

with DOCUMENTS_FILE.open("r", encoding="utf-8")as file:
    documents =json.load(file)


with QUESTION_FILE.open("r", encoding="utf-8")as file:
    questions= json.load(file)



experiment_documents =[
    document for document in documents
    if document["source"] == "experiment_corpus.pdf"

]

#Check whether the expanded dataset is ready

for document in experiment_documents:
    token_count = len(tokenizer.encode(document["text"]))

    if token_count <= 1024:
        raise ValueError(
             f"Page {document['page']} has only {token_count} tokens. "
            "Please check the expanded PDF and extract it again."   
        )

print("Dataset validation passed!")
print(f"Pages available: {len(experiment_documents)}")
print(f"Evaluation questions: {len(questions)}")


# Split text into token chunks
def create_chunks(text, chunk_size, overlap):
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        chunks.append(tokenizer.decode(chunk_tokens))

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks

#Load embedding model once
print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully!")


results = []

for chunk_size in CHUNK_SIZES:

    print(f"\nTesting chunk size: {chunk_size}")

    all_chunks = []

    # Create chunks for every page
    for document in experiment_documents:

        chunks = create_chunks(
            document["text"],
            chunk_size,
            OVERLAP
        )

        for chunk_index, chunk_text in enumerate(chunks):

            all_chunks.append({
                "source": document["source"],
                "page": document["page"],
                "chunk_index": chunk_index,
                "text": chunk_text
            })

    # Generate embeddings
    chunk_texts = [
        chunk["text"] for chunk in all_chunks
    ]

    chunk_embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_texts = [
        item["question"] for item in questions
    ]

    query_embeddings = model.encode(
        query_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Evaluate retrieval
    hit_counts = {1: 0, 3: 0, 5: 0}

    for question_index, question in enumerate(questions):

        similarities = cosine_similarity(
            query_embeddings[question_index].reshape(1, -1),
            chunk_embeddings
        )[0]

        ranked_indices = np.argsort(similarities)[::-1]

        expected_page = question["expected_page"]

        for k in [1, 3, 5]:

            top_results = ranked_indices[:k]

            retrieved_pages = [
                all_chunks[index]["page"]
                for index in top_results
            ]

            if expected_page in retrieved_pages:
                hit_counts[k] += 1

    # Calculate hit rates
    total_questions = len(questions)

    hit_rates = {
        k: round((hit_counts[k] / total_questions) * 100, 2)
        for k in [1, 3, 5]
    }

    results.append({
        "chunk_size": chunk_size,
        "total_chunks": len(all_chunks),
        "hit_at_1": hit_rates[1],
        "hit_at_3": hit_rates[3],
        "hit_at_5": hit_rates[5]
    })


# Display comparison table
print("\nCHUNK SIZE EXPERIMENT RESULTS")
print("-" * 70)

print(
    f"{'Chunk Size':<15}"
    f"{'Total Chunks':<15}"
    f"{'Hit@1':<12}"
    f"{'Hit@3':<12}"
    f"{'Hit@5':<12}"
)

print("-" * 70)

for result in results:
    print(
        f"{result['chunk_size']:<15}"
        f"{result['total_chunks']:<15}"
        f"{result['hit_at_1']:<12}"
        f"{result['hit_at_3']:<12}"
        f"{result['hit_at_5']:<12}"
    )


# Save results
output_file = OUTPUT_DIR / "chunk_size_results.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print(f"\nResults saved to: {output_file}")
