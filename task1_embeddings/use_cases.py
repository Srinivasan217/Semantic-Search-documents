
import numpy as np

from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from common import SENTENCES, CATEGORIES, load_model


# Load model and generate embeddings
model = load_model()

embeddings = model.encode(
    SENTENCES,
    convert_to_numpy=True
)


# ==================================
# USE CASE 1: SEMANTIC SEARCH
# ==================================

print("\n========== SEMANTIC SEARCH ==========")

queries = [
    "How can employees update their login credentials?",
    "How do computers learn from information?",
    "How can workers book a meeting room?"
]

for query in queries:

    query_embedding = model.encode([query])

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:3]

    print("\nQuery:", query)

    for index in top_indices:
        print(
            f"Score: {scores[index]:.4f} | "
            f"{SENTENCES[index]}"
        )


# ==================================
# USE CASE 2: DUPLICATE DETECTION
# ==================================

print("\n========== DUPLICATE DETECTION ==========")

threshold = 0.70

similarity_matrix = cosine_similarity(embeddings)

print("Duplicate threshold:", threshold)

found = False

for i in range(len(SENTENCES)):
    for j in range(i + 1, len(SENTENCES)):

        score = similarity_matrix[i][j]

        if score >= threshold:
            found = True

            print(f"\nSimilarity: {score:.4f}")
            print("Sentence 1:", SENTENCES[i])
            print("Sentence 2:", SENTENCES[j])
            print("Possible duplicate detected!")

if not found:
    print("No sentence pairs exceeded the threshold.")


# ==================================
# USE CASE 3: K-MEANS CLUSTERING
# ==================================

print("\n========== K-MEANS CLUSTERING ==========")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(embeddings)

for cluster_id in sorted(set(cluster_labels)):

    members = [
        i for i, label in enumerate(cluster_labels)
        if label == cluster_id
    ]

    # Assign a descriptive name based on
    # the majority category in this cluster
    category_counts = Counter(
        CATEGORIES[i] for i in members
    )

    cluster_name = category_counts.most_common(1)[0][0]

    print(f"\nCluster {cluster_id + 1}: {cluster_name}")

    for index in members:
        print("-", SENTENCES[index])

print("\nAll three use cases completed!")
