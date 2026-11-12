import numpy as np

from sklearn.metrics.pairwise import cosine_similarity as sklearn_similarity

def cosine_similarity(a,b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)


    dot_product = np.dot(a,b)

    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


#Test cases
test_cases = [
    ([1,2,3], [2,4,6]),
    ([1,0],[0,1]),
    ([1,0],[-1, 0])
]

print("--- MANUAL COSINE SIMILARITY ---")

for a ,b in test_cases:
    result = cosine_similarity(a,b)

    print(f"{a} vs {b} -> {result:1f}")

print("\n --- SCIKIT-LEARN COMPARISON ---")

for a,b in test_cases:
    manual = cosine_similarity(a,b)

    sklearn_result = sklearn_similarity(
        [a],[b]
    )[0][0]

    print(f"Manual: {manual:.4f}")
    print(f"Scikit-learn: {sklearn_result:.4f}")
    print("Match:", np.isclose(manual, sklearn_result))
    print()