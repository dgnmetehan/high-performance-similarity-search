"""
Phase 1: Naive Baseline Search

This is your starting point -- intentionally slow pure Python.
Your job: profile it, understand why it's slow, then optimise.

The story starts here: you have a corpus and queries. You need top-k similar docs.
"""

import math
import time
from pathlib import Path

import numpy as np


def cosine_similarity_python(a, b):
    """
    Compute cosine similarity between two vectors using pure Python.
    NO NumPy inside this function -- this is the bottleneck you'll fix later.
    """
    # TODO: implement pure Python cosine similarity
    # Hint: dot product / (norm_a * norm_b)
    
    dot_product = 0.0
    sum_a_squared = 0.0
    sum_b_squared = 0.0

    for i in range(len(a)):
        dot_product += a[i] * b[i]
        sum_a_squared += a[i] * a[i]
        sum_b_squared += b[i] * b[i]

    norm_a = math.sqrt(sum_a_squared)
    norm_b = math.sqrt(sum_b_squared)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def search_naive(queries, corpus, k=5):
    """
    For each query, find the top-k most similar documents in corpus.

    Args:
        queries: list of lists (or list of arrays) -- query vectors
        corpus: list of lists (or list of arrays) -- document vectors
        k: number of top results to return

    Returns:
        list of lists: top-k (doc_index, similarity_score) for each query
   
   Find the top-k most similar documents for each query."""
    results = []

    for query in queries:
        scores = []

        for doc_index, doc in enumerate(corpus):
            similarity = cosine_similarity_python(query, doc)
            scores.append(
                (doc_index, similarity))

        scores.sort(key=lambda item: item[1], reverse=True)
        results.append(scores[:k])

    return results


if __name__ == "__main__":

    # Load your dataset here
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    CORPUS_SIZE = 12000

    EMBEDDINGS_DIR = PROJECT_DIR / "embeddings" / "12k"
    RESULTS_DIR = PROJECT_DIR / "results" / "phase1" / "12k"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")

    try:
        corpus = np.load(
            EMBEDDINGS_DIR / "corpus_embeddings.npy"
        ).tolist()

        queries = np.load(
            EMBEDDINGS_DIR / "query_embeddings.npy"
        ).tolist()

    except FileNotFoundError as exc:
        print(f"Embedding file not found: {exc.filename}")
        print("Check your embedding files first.")
        raise SystemExit(1)

    if not corpus or not queries:
        raise SystemExit("Corpus and queries must not be empty.")

    if len(corpus) != CORPUS_SIZE:
        raise SystemExit(
            f"Expected {CORPUS_SIZE} documents, found {len(corpus)}."
        )

    print(
        f"Corpus: {len(corpus)} documents x "
        f"{len(corpus[0])} dimensions"
    )
    print(f"Queries: {len(queries)} queries")

    # Baseline benchmark
    print("\nRunning naive baseline...")

    t0 = time.perf_counter()
    results = search_naive(queries, corpus, k=5)
    t1 = time.perf_counter()

    baseline_time = t1 - t0

    print(f"\nBaseline time: {baseline_time:.3f} seconds")
    print(
        f"Time per query: "
        f"{baseline_time / len(queries) * 1000:.1f} ms"
    )
    print(f"First query top-5: {results[0]}")

    # Save baseline time for comparison in later phases
    baseline_time_file = RESULTS_DIR / "baseline_time.txt"

    with open(baseline_time_file, "w", encoding="utf-8") as file:
        file.write(f"{baseline_time}\n")

    print(f"Saved baseline time: {baseline_time_file}")


