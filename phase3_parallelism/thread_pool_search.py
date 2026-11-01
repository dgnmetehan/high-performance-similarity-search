"""
Phase 3: ThreadPoolExecutor Strategy

When to use this: I/O-bound tasks or NumPy/BLAS operations (GIL released).

The story: you've vectorised your search. Now you have multiple queries.
Can you process them in parallel with threads?

Key question: does the GIL block you, or does BLAS release it?
"""
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = PROJECT_DIR / "embeddings" / "12k"
RESULTS_DIR = PROJECT_DIR / "results" / "phase3"

sys.path.insert(0, str(PROJECT_DIR))

from benchmark_utils import (
    benchmark,
    verify_results,
    print_comparison_table,
    save_results_csv,
    plot_speedup,
    plot_scaling,
    plot_amdahl,
)


def normalize_embeddings(embeddings):
    """L2-normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def search_single_numpy(query, corpus, k=5):
    """Search one query using NumPy vectorisation."""
    similarities = query @ corpus.T

    top_k_idx = np.argpartition(-similarities, k - 1)[:k]
    top_k_scores = similarities[top_k_idx]

    sorted_order = np.argsort(-top_k_scores)

    return list(zip(
        top_k_idx[sorted_order],
        top_k_scores[sorted_order]
    ))


def search_numpy_serial(queries, corpus, k=5):
    """Process all queries sequentially using NumPy."""
    return [
        search_single_numpy(query, corpus, k)
        for query in queries
    ]


def search_threadpool(queries, corpus, k=5, max_workers=4):
    """
    Process queries in parallel using ThreadPoolExecutor.
    Each query gets its own thread.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_single_numpy, query, corpus, k)
            for query in queries
        ]

        return [future.result() for future in futures]


def benchmark_thread_scaling(queries, corpus, k=5):
    """
    Benchmark with 1, 2, 4, 8 workers.
    Plot speedup vs. workers to see if scaling is linear.
    """
    worker_counts = [1, 2, 4, 8]
    results = {}

    # Serial NumPy reference
    print("\n--- Serial NumPy ---")
    results["Serial NumPy"] = benchmark(
        search_numpy_serial,
        queries,
        corpus,
        k,
        warmup=3,
        runs=5
    )

    print(
        f"Mean: {results['Serial NumPy']['mean']:.6f}s, "
        f"Std: {results['Serial NumPy']['std']:.6f}s"
    )

    # ThreadPool experiments
    for workers in worker_counts:
        name = f"ThreadPool {workers}"

        print(f"\n--- {name} ---")

        results[name] = benchmark(
            search_threadpool,
            queries,
            corpus,
            k,
            workers,
            warmup=3,
            runs=5
        )

        print(
            f"Mean: {results[name]['mean']:.6f}s, "
            f"Std: {results[name]['std']:.6f}s"
        )

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: ThreadPoolExecutor Strategy")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    corpus = np.load(EMBEDDING_DIR / "corpus_embeddings.npy")
    queries = np.load(EMBEDDING_DIR / "query_embeddings.npy")

    print(f"Corpus shape: {corpus.shape}")
    print(f"Queries shape: {queries.shape}")
    print(f"CPU cores: {os.cpu_count()}")

    # Normalize outside benchmark
    corpus = normalize_embeddings(corpus)
    queries = normalize_embeddings(queries)

    print("\nEmbeddings normalized for cosine similarity.")

    # Run benchmarks
    results = benchmark_thread_scaling(queries, corpus, k=5)

    # Print comparison table
    print_comparison_table(
        results,
        baseline_name="Serial NumPy"
    )

    # Save timing results
    save_results_csv(
        results,
        RESULTS_DIR / "threadpool_benchmark.csv"
    )

    # Calculate speedups relative to Serial NumPy
    serial_time = results["Serial NumPy"]["mean"]

    worker_counts = [1, 2, 4, 8]
    worker_times = [
        results[f"ThreadPool {w}"]["mean"]
        for w in worker_counts
    ]

    overall_speedups = [
        serial_time / t for t in worker_times
    ]

    # Calculate speedups relative to ThreadPool 1 worker
    single_thread_time = results["ThreadPool 1"]["mean"]

    thread_scaling_speedups = [
        single_thread_time / t for t in worker_times
    ]

    print("\nSpeedup vs Serial NumPy:")
    for workers, speedup in zip(worker_counts, overall_speedups):
        print(f"{workers} workers: {speedup:.3f}x")

    print("\nThread scaling speedup vs ThreadPool 1:")
    for workers, speedup in zip(worker_counts, thread_scaling_speedups):
        print(f"{workers} workers: {speedup:.3f}x")

    # Plot 1: Overall speedup vs Serial NumPy
    plot_speedup(
        names=["Serial NumPy"] +
              [f"ThreadPool {w}" for w in worker_counts],
        times=[serial_time] + worker_times,
        baseline_name="Serial NumPy",
        title="ThreadPool Speedup vs Serial NumPy",
        save_path=str(RESULTS_DIR / "threadpool_speedup.png")
    )

    # Plot 2: Scaling with worker count
    plot_scaling(
        x_values=worker_counts,
        y_values_dict={
            "Speedup vs Serial NumPy": overall_speedups,
            "Scaling vs ThreadPool 1": thread_scaling_speedups,
        },
        xlabel="Number of Workers",
        ylabel="Speedup",
        title="ThreadPool Scaling Performance",
        save_path=str(RESULTS_DIR / "threadpool_scaling.png")
    )

    # Plot 3: Amdahl's Law comparison
    plot_amdahl(
        worker_counts=worker_counts,
        speedups=thread_scaling_speedups,
        title="ThreadPool Scaling vs Amdahl's Law",
        save_path=str(RESULTS_DIR / "threadpool_amdahl.png")
    )

    # Correctness verification
    reference = results["Serial NumPy"]["results"]

    print("\nCorrectness Check:")
    for workers in worker_counts:
        parallel_results = results[f"ThreadPool {workers}"]["results"]

        correct = verify_results(
            reference,
            parallel_results,
            tol=1e-5
        )

        print(
            f"ThreadPool {workers}: "
            f"{'PASS' if correct else 'FAIL'}"
        )

    print("\nAll outputs saved to:", RESULTS_DIR)
    print("=" * 60)
