
"""
Phase 3: Async I/O Strategy

When to use this: massive concurrent I/O (>100 tasks, e.g., batch API queries).

The story: your search is now fast. But what if queries come in
as async HTTP requests? Or what if you need to call an embedding API?

Key question: does async help CPU-bound search, or only I/O-bound retrieval?
"""

import numpy as np
import time
import asyncio
import sys
import csv
import hashlib
import matplotlib.pyplot as plt

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from benchmark_utils import benchmark


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = PROJECT_DIR / "embeddings" / "12k"
RESULTS_DIR = PROJECT_DIR / "results" / "phase3"

CORPUS_PATH = EMBEDDING_DIR / "corpus_embeddings.npy"
QUERIES_PATH = EMBEDDING_DIR / "query_embeddings.npy"


async def get_embedding_async(text, delay=0.01, embedding_dim=384):
    """
    Simulate an async embedding API call.
    In a real system, this would be an HTTP request to an embedding service.
    """
    await asyncio.sleep(delay)  # network latency

    # Generate a reproducible embedding for each query
    seed = int.from_bytes(
        hashlib.sha256(text.encode()).digest()[:4],
        byteorder="little"
    )

    rng = np.random.default_rng(seed)
    return rng.standard_normal(embedding_dim).astype(np.float32)


async def search_api_async(query_text, corpus, k=5, embed_delay=0.01):
    """
    Full async pipeline:
    1. Async embedding retrieval (I/O)
    2. CPU-bound similarity (blocks event loop!)
    3. Return top-k
    """
    # Step 1: async I/O
    query_embedding = await get_embedding_async(
        query_text,
        delay=embed_delay,
        embedding_dim=corpus.shape[1]
    )

    # Step 2: CPU work (this blocks!)
    similarities = query_embedding @ corpus.T
    top_k_idx = np.argpartition(-similarities, k - 1)[:k]
    top_k_scores = similarities[top_k_idx]
    sorted_order = np.argsort(-top_k_scores)

    return list(zip(top_k_idx[sorted_order], top_k_scores[sorted_order]))


async def search_all_serial(query_texts, corpus, k=5, embed_delay=0.01):
    """Process queries one at a time (serial async)."""
    results = []
    for text in query_texts:
        result = await search_api_async(text, corpus, k, embed_delay)
        results.append(result)
    return results


async def search_all_concurrent(query_texts, corpus, k=5, embed_delay=0.01):
    """Process all queries concurrently with asyncio.gather."""
    tasks = [
        search_api_async(text, corpus, k, embed_delay)
        for text in query_texts
    ]
    return await asyncio.gather(*tasks)


async def search_all_bounded(query_texts, corpus, k=5, embed_delay=0.01, max_concurrent=10):
    """
    Bounded concurrency with semaphore.
    Prevents overwhelming the API.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_search(text):
        async with semaphore:
            return await search_api_async(text, corpus, k, embed_delay)

    tasks = [bounded_search(text) for text in query_texts]
    return await asyncio.gather(*tasks)


def search_numpy_single(query, corpus, k=5):
    """Search one query using NumPy."""
    similarities = query @ corpus.T
    top_indices = np.argpartition(-similarities, k - 1)[:k]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]

    return top_indices, similarities[top_indices]


def search_precomputed_serial(queries, corpus, k=5):
    """Search pre-computed embeddings sequentially."""
    return [
        search_numpy_single(query, corpus, k)
        for query in queries
    ]


async def search_precomputed_async(query, corpus, k=5):
    """Async wrapper for CPU-bound search with pre-computed embeddings."""
    return search_numpy_single(query, corpus, k)


async def search_all_precomputed_serial(queries, corpus, k=5):
    """Process pre-computed queries one at a time."""
    results = []

    for query in queries:
        result = await search_precomputed_async(query, corpus, k)
        results.append(result)

    return results


async def search_all_precomputed_concurrent(queries, corpus, k=5):
    """Process pre-computed queries with asyncio.gather."""
    tasks = [
        search_precomputed_async(query, corpus, k)
        for query in queries
    ]

    return await asyncio.gather(*tasks)


async def search_all_precomputed_threadpool(queries, corpus, k=5, max_workers=4):
    """Combine async execution with ThreadPoolExecutor for CPU work."""
    loop = asyncio.get_running_loop()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                search_numpy_single,
                query,
                corpus,
                k
            )
            for query in queries
        ]

        return await asyncio.gather(*tasks)


def run_async_serial(query_texts, corpus, k=5, embed_delay=0.01):
    return asyncio.run(
        search_all_serial(query_texts, corpus, k, embed_delay)
    )


def run_async_concurrent(query_texts, corpus, k=5, embed_delay=0.01):
    return asyncio.run(
        search_all_concurrent(query_texts, corpus, k, embed_delay)
    )


def run_async_bounded(query_texts, corpus, k=5, embed_delay=0.01):
    return asyncio.run(
        search_all_bounded(
            query_texts,
            corpus,
            k,
            embed_delay,
            max_concurrent=10
        )
    )


def run_precomputed_serial(queries, corpus, k=5):
    return search_precomputed_serial(queries, corpus, k)


def run_precomputed_async_serial(queries, corpus, k=5):
    return asyncio.run(
        search_all_precomputed_serial(queries, corpus, k)
    )


def run_precomputed_async_concurrent(queries, corpus, k=5):
    return asyncio.run(
        search_all_precomputed_concurrent(queries, corpus, k)
    )


def run_precomputed_threadpool(queries, corpus, k=5, max_workers=4):
    return asyncio.run(
        search_all_precomputed_threadpool(
            queries,
            corpus,
            k,
            max_workers
        )
    )


def benchmark_async(query_texts, corpus, k=5, embed_delay=0.01):
    """Benchmark serial vs concurrent vs bounded async."""
    results = {}

    methods = {
        "serial": run_async_serial,
        "concurrent": run_async_concurrent,
        "bounded": run_async_bounded
    }

    for name, func in methods.items():
        benchmark_result = benchmark(
            func,
            query_texts,
            corpus,
            k=k,
            embed_delay=embed_delay,
            warmup=1,
            runs=3
        )

        results[name] = benchmark_result

        print(
            f"{name.capitalize():<12} "
            f"Mean: {benchmark_result['mean']:.4f}s, "
            f"Std: {benchmark_result['std']:.4f}s"
        )

    return results


def benchmark_precomputed(queries, corpus, k=5):
    """Benchmark CPU-bound search without embedding API latency."""
    results = {}

    methods = {
        "numpy_serial": run_precomputed_serial,
        "async_serial": run_precomputed_async_serial,
        "async_concurrent": run_precomputed_async_concurrent,
        "async_threadpool": run_precomputed_threadpool
    }

    for name, func in methods.items():
        benchmark_result = benchmark(
            func,
            queries,
            corpus,
            k=k,
            warmup=3,
            runs=5
        )

        results[name] = benchmark_result

        print(
            f"{name:<20} "
            f"Mean: {benchmark_result['mean']:.6f}s, "
            f"Std: {benchmark_result['std']:.6f}s"
        )

    return results


def save_results_csv(api_results, precomputed_results, output_path):
    """Save async and pre-computed benchmark results to CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Experiment",
            "API Delay (s)",
            "Method",
            "Mean Time (s)",
            "Std Time (s)"
        ])

        for delay, methods in api_results.items():
            for name, result in methods.items():
                writer.writerow([
                    "API latency",
                    delay,
                    name,
                    result["mean"],
                    result["std"]
                ])

        for name, result in precomputed_results.items():
            writer.writerow([
                "Pre-computed embeddings",
                0,
                name,
                result["mean"],
                result["std"]
            ])


def plot_api_scaling(api_results, output_path):
    """Plot API concurrency speedup against serial execution."""
    delays = list(api_results.keys())
    methods = ["concurrent", "bounded"]

    x = np.arange(len(delays))
    width = 0.3

    plt.figure(figsize=(8, 5))

    for i, method in enumerate(methods):
        speedups = [
            api_results[delay]["serial"]["mean"] /
            api_results[delay][method]["mean"]
            for delay in delays
        ]

        plt.bar(
            x + (i - 0.5) * width,
            speedups,
            width,
            label=method.capitalize()
        )

    plt.axhline(1.0, linestyle="--", label="Serial baseline")

    plt.xticks(
        x,
        [f"{delay:.2f}s" for delay in delays]
    )

    plt.xlabel("Simulated API Latency")
    plt.ylabel("Speedup vs Serial")
    plt.title("Async API Concurrency Performance")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_precomputed_performance(precomputed_results, output_path):
    """Plot execution time for pre-computed embedding searches."""
    names = list(precomputed_results.keys())

    times = [
        precomputed_results[name]["mean"]
        for name in names
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(names, times)

    plt.ylabel("Execution Time (seconds)")
    plt.xlabel("Search Strategy")
    plt.title("Pre-computed Embeddings: CPU Search Performance")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: Async I/O Strategy")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        corpus = np.load(CORPUS_PATH)
        queries = np.load(QUERIES_PATH)
    except FileNotFoundError:
        print("Embeddings not found. Generate your dataset first!")
        sys.exit(1)

    # Normalize embeddings for cosine similarity
    corpus = corpus.astype(np.float32)
    queries = queries.astype(np.float32)

    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)

    corpus = corpus / np.maximum(corpus_norms, 1e-12)
    queries = queries / np.maximum(query_norms, 1e-12)

    query_texts = [f"Query about ML topic {i}" for i in range(100)]

    print(f"Corpus: {corpus.shape}")
    print(f"Pre-computed queries: {queries.shape}")
    print(f"Simulated API queries: {len(query_texts)}")

    # 1. Does async help when the API latency is the bottleneck?
    print("\n--- API latency experiment ---")

    api_results = {}

    for delay in [0.01, 0.05]:
        print(f"\n--- Simulated API latency: {delay}s ---")

        api_results[delay] = benchmark_async(
            query_texts,
            corpus,
            k=5,
            embed_delay=delay
        )

        speedup_concurrent = (
            api_results[delay]["serial"]["mean"] /
            api_results[delay]["concurrent"]["mean"]
        )

        speedup_bounded = (
            api_results[delay]["serial"]["mean"] /
            api_results[delay]["bounded"]["mean"]
        )

        print(f"Concurrent speedup: {speedup_concurrent:.2f}x")
        print(f"Bounded speedup: {speedup_bounded:.2f}x")

    # 2. What if the embedding is pre-computed (no I/O)?
    print("\n--- Pre-computed embeddings experiment ---")

    precomputed_results = benchmark_precomputed(
        queries,
        corpus,
        k=5
    )

    # 3. When would bounded concurrency matter?
    print("\n--- Bounded concurrency comparison ---")

    for delay, methods in api_results.items():
        print(
            f"API delay {delay:.2f}s | "
            f"Concurrent: {methods['concurrent']['mean']:.4f}s | "
            f"Bounded: {methods['bounded']['mean']:.4f}s"
        )

    # 4. How would you combine async I/O with ThreadPool for CPU work?
    print("\n--- Async + ThreadPool experiment ---")

    print(
        "Async + ThreadPool time: "
        f"{precomputed_results['async_threadpool']['mean']:.6f}s"
    )

    # Save results and plots
    csv_path = RESULTS_DIR / "async_benchmark.csv"
    api_graph_path = RESULTS_DIR / "async_api_scaling.png"
    precomputed_graph_path = RESULTS_DIR / "async_precomputed_performance.png"

    save_results_csv(
        api_results,
        precomputed_results,
        csv_path
    )

    plot_api_scaling(
        api_results,
        api_graph_path
    )

    plot_precomputed_performance(
        precomputed_results,
        precomputed_graph_path
    )

    print("\nResults saved:")
    print(csv_path)
    print(api_graph_path)
    print(precomputed_graph_path)

    print("\n" + "=" * 60)
    print("Async I/O experiments completed.")
    print("=" * 60)
