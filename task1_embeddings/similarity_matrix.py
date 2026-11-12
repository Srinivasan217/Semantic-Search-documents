import re
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from common import SENTENCES, load_model


#Load local model
model = load_model()

embeddings = model.encode(
    SENTENCES,
    convert_to_numpy=True
)

# Calculate similarity matrix
matrix = cosine_similarity(embeddings)

print("\n--- SIMILARITY MATRIX ---")

print(np.round(matrix, 3))

# Find the three most similar pairs
pairs = []

for i in range(len(SENTENCES)):
    for j in range(i + 1, len(SENTENCES)):
        pairs.append((
            matrix[i][j],
            i,
            j
        ))

pairs.sort(reverse=True)

print("\n--- TOP 3 MOST SIMILAR PAIRS ---")

for score, i, j in pairs[:3]:
    print(f"\nSimilarity: {score:.4f}")
    print("Sentence 1:", SENTENCES[i])
    print("Sentence 2:", SENTENCES[j])


# Demonstrate semantic similarity without common words
def tokenize(text):
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


i, j = 4, 5

common_words = tokenize(SENTENCES[i]) & tokenize(SENTENCES[j])

print("\n--- NO COMMON WORDS EXAMPLE ---")
print("Sentence 1:", SENTENCES[i])
print("Sentence 2:", SENTENCES[j])
print("Common words:", common_words)
print("Similarity:", round(matrix[i][j], 4))

if not common_words:
    print("These sentences share no exact words.")
    print("Their embedding similarity demonstrates semantic matching.")
