import os
import time
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from common import SENTENCES, load_model

#Load API key from the project's root .env file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(exist_ok=True)

#1.LOCAL EMBEDDINGS

print("\n--- LOCAL EMBEDDING MODEL ---")

model = load_model()

start_time = time.perf_counter()

local_embeddings = model.encode(
    SENTENCES,
    convert_to_numpy=True
)

local_time = time.perf_counter() - start_time


print("Model: all-MiniLM-L6-v2")
print("Number of sentences:", len(SENTENCES))

print("Embedding shape:", local_embeddings.shape)
print("Embedding dimensions:", local_embeddings.shape[1])
print(f"Local execution time: {local_time:.4f} seconds")

np.save(OUTPUT_DIR / "local_embeddings.npy", local_embeddings)

#2. API EMBEDDINGS

print("\n--- OPENROUTER API EMBEDDINGS ---")

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is missing."
        "Add it to the project's root .env file."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

start_time = time.perf_counter()
response = client.embeddings.create(
    model="openai/text-embedding-3-small",
    input=SENTENCES
)

api_time = time.perf_counter() - start_time

#Preserve the original sentence order
api_embeddings = np.array([
    item.embedding for item in
    sorted(response.data,key=lambda item:item.index)
])

print("Model: openai/text-embedding-3-small")
print("Number of sentences:", len(SENTENCES))
print("Embedding shape:", api_embeddings.shape)
print("Embedding dimensions:", api_embeddings.shape[1])
print(f"API execution time: {api_time:.4f} seconds")

np.save(OUTPUT_DIR / "api_embeddings.npy", api_embeddings)

#3. COMPARISON

print("\n --- PERFORMANCE COMPARISON ---")

print(f"Local model time: {local_time:.4f} seconds")
print(f"API model time:   {api_time:.4f} seconds")

if local_time < api_time:
    print("Local model was faster in this run.")
else:
    print("API model was faster in this run.")

print("\nEmbedding comparison completed!")