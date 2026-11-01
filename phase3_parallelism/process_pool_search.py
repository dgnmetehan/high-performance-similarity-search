
"""
Phase 3: ProcessPoolExecutor Strategy

When to use this: CPU-bound pure Python (GIL blocks threads).

The story: threads didn't help because of the GIL.
Processes bypass the GIL entirely. But there's a cost: pickling.

Key question: how do you share data between processes without
paying the pickling penalty?
"""

import numpy as np
import time
import csv
import os
import sys
import matplotlib.pyplot as plt

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import shared_memory

sys.path.append(str(Path(__file__).resolve().parent.parent))
from benchmark_utils import benchmark


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = PROJECT_DIR / "embeddings" / "12k"
RESULTS_DIR = PROJECT_DIR / "results" / "phase3"

CORPUS_PATH = EMBEDDING_DIR / "corpus_embeddings.npy"
QUERIES_PATH = EMBEDDING_DIR / "query_embeddings.npy"

# Shared memory variables used by worker processes
_worker_corpus = None
_worker_queries = None
_worker_corpus_shm = None
_worker_queries_shm = None


def search_batch_python(query_batch, corpus_list, k=5):
    """
    Search a batch of queries using pure Python.
    This function runs in a separate process.
    """
    results = []
    for query in query_batch:
        scores = []
        for d_idx, doc in enumerate(corpus_list):
            # Pure Python cosine similarity
            dot = 0.0
            norm_q = 0.0
            norm_d = 0.0
            for i in range(len(query)):
                dot += query[i] * doc[i]
                norm_q += query[i] * query[i]
                norm_d += doc[i] * doc[i]
            sim = dot / (norm_q ** 0.5 * norm_d ** 0.5) if norm_q > 0 and norm_d > 0 else 0.0
            scores.append((d_idx, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        results.append(scores[:k])
    return results


def search_processpool_bad(queries_list, corpus_list, k=5, max_workers=4):
    """
    BAD approach: submit one task per query, pass corpus every time.
    This pickles the entire corpus for each query. Very slow.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_batch_python, [queries_list[i]], corpus_list, k)
            for i in range(len(queries_list))
        ]
        results = [f.result()[0] for f in futures]
    return results


def search_processpool_good(queries_list, corpus_list, k=5, max_workers=4):
    """
    GOOD approach: pass indices, not data. Or batch queries per process.
    Each worker gets a chunk of queries, reducing pickling overhead.
    """
    # Split queries into batches
    batch_size = (len(queries_list) + max_workers - 1) // max_workers
    batches = [
        queries_list[i:i + batch_size]
        for i in range(0, len(queries_list), batch_size)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_batch_python, batch, corpus_list, k)
            for batch in batches
        ]
        results = []
        for f in futures:
            results.extend(f.result())
    return results


def search_numpy_single(query, corpus, k=5):
    """Search one query using NumPy."""
    similarities = query @ corpus.T
    top_indices = np.argpartition(-similarities, k - 1)[:k]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]

    return top_indices, similarities[top_indices]


def search_numpy_serial(queries, corpus, k=5):
    """Sequential NumPy search used as the baseline."""
    return [
        search_numpy_single(query, corpus, k)
        for query in queries
    ]


def search_threadpool_numpy(queries, corpus, k=5, max_workers=4):
    """Run NumPy similarity searches concurrently using threads."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(search_numpy_single, query, corpus, k)
            for query in queries
        ]
        return [future.result() for future in futures]


def init_shared_memory(corpus_name, corpus_shape, corpus_dtype,
                       queries_name, queries_shape, queries_dtype):
    """Attach worker processes to the shared memory arrays."""
    global _worker_corpus
    global _worker_queries
    global _worker_corpus_shm
    global _worker_queries_shm

    _worker_corpus_shm = shared_memory.SharedMemory(name=corpus_name)
    _worker_queries_shm = shared_memory.SharedMemory(name=queries_name)

    _worker_corpus = np.ndarray(
        corpus_shape,
        dtype=np.dtype(corpus_dtype),
        buffer=_worker_corpus_shm.buf
    )

    _worker_queries = np.ndarray(
        queries_shape,
        dtype=np.dtype(queries_dtype),
        buffer=_worker_queries_shm.buf
    )


def search_shared_memory_range(start, end, k):
    """Search a range of query indices using shared memory."""
    results = []

    query_batch = _worker_queries[start:end]
    similarities = query_batch @ _worker_corpus.T

    for row in similarities:
        top_indices = np.argpartition(-row, k - 1)[:k]
        top_indices = top_indices[np.argsort(-row[top_indices])]
        results.append((top_indices, row[top_indices]))

    return results


def search_processpool_numpy(queries, corpus, k=5, max_workers=4):
    """
    ProcessPool with NumPy and shared memory.
    Corpus and queries are copied to shared memory once.
    Workers receive query index ranges instead of embedding arrays.
    """
    queries = np.ascontiguousarray(queries)
    corpus = np.ascontiguousarray(corpus)

    corpus_shm = shared_memory.SharedMemory(
        create=True,
        size=corpus.nbytes
    )

    queries_shm = shared_memory.SharedMemory(
        create=True,
        size=queries.nbytes
    )

    try:
        shared_corpus = np.ndarray(
            corpus.shape,
            dtype=corpus.dtype,
            buffer=corpus_shm.buf
        )
        shared_queries = np.ndarray(
            queries.shape,
            dtype=queries.dtype,
            buffer=queries_shm.buf
        )

        shared_corpus[:] = corpus
        shared_queries[:] = queries

        batch_size = (len(queries) + max_workers - 1) // max_workers
        ranges = [
            (start, min(start + batch_size, len(queries)))
            for start in range(0, len(queries), batch_size)
        ]

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=init_shared_memory,
            initargs=(
                corpus_shm.name,
                corpus.shape,
                corpus.dtype.str,
                queries_shm.name,
                queries.shape,
                queries.dtype.str
            )
        ) as executor:

            futures = [
                executor.submit(search_shared_memory_range, start, end, k)
                for start, end in ranges
            ]

            results = []
            for future in futures:
                results.extend(future.result())

        return results

    finally:
        corpus_shm.close()
        corpus_shm.unlink()

        queries_shm.close()
        queries_shm.unlink()


def verify_results(reference, candidate, tol=1e-5):
    """Check whether top-k indices and similarity scores match."""
    if len(reference) != len(candidate):
        return False

    for ref, result in zip(reference, candidate):
        ref_indices, ref_scores = ref
        indices, scores = result

        if not np.array_equal(ref_indices, indices):
            return False

        if not np.allclose(ref_scores, scores, atol=tol, rtol=tol):
            return False

    return True


def save_benchmark_csv(results, output_path):
    """Save benchmark measurements and speedups to CSV."""
    baseline_time = results["Sequential NumPy"]["mean"]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Method",
            "Workers",
            "Mean Time (s)",
            "Std Time (s)",
            "Min Time (s)",
            "Max Time (s)",
            "Speedup vs Sequential NumPy"
        ])

        for name, data in results.items():
            speedup = baseline_time / data["mean"]

            writer.writerow([
                name,
                data["workers"],
                data["mean"],
                data["std"],
                data["min"],
                data["max"],
                speedup
            ])


def plot_scaling(results, output_path):
    """Plot speedup against the number of workers."""
    baseline_time = results["Sequential NumPy"]["mean"]
    workers = [1, 2, 4, 8]

    process_speedups = [
        baseline_time / results[f"ProcessPool {w}"]["mean"]
        for w in workers
    ]

    thread_speedups = [
        baseline_time / results[f"ThreadPool {w}"]["mean"]
        for w in workers
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        workers,
        process_speedups,
        marker="o",
        label="ProcessPool + Shared Memory"
    )

    plt.plot(
        workers,
        thread_speedups,
        marker="s",
        label="ThreadPool"
    )

    plt.axhline(
        1.0,
        linestyle="--",
        label="Sequential NumPy baseline"
    )

    plt.xticks(workers)
    plt.xlabel("Number of Workers")
    plt.ylabel("Speedup vs Sequential NumPy")
    plt.title("ProcessPool and ThreadPool Scaling Performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: ProcessPoolExecutor Strategy")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        corpus = np.load(CORPUS_PATH)
        queries = np.load(QUERIES_PATH)
    except FileNotFoundError:
        print("Embeddings not found. Generate your dataset first!")
        sys.exit(1)

    print(f"Corpus: {corpus.shape}, Queries: {queries.shape}")

    # Normalize embeddings for cosine similarity
    corpus = corpus.astype(np.float32)
    queries = queries.astype(np.float32)

    corpus_norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)

    corpus = corpus / np.maximum(corpus_norms, 1e-12)
    queries = queries / np.maximum(query_norms, 1e-12)

    print("Embeddings normalized for cosine similarity.")

    # Compare bad vs good approach on a smaller sample
    print("\n--- Bad approach (pickle per query) ---")

    small_queries = queries[:2].tolist()
    small_corpus = corpus[:500].tolist()

    t0 = time.perf_counter()
    _ = search_processpool_bad(
        small_queries,
        small_corpus,
        k=5,
        max_workers=4
    )
    t1 = time.perf_counter()

    print(f"Bad approach: {t1 - t0:.3f}s (2 queries, 500 documents)")

    print("\n--- Good approach (batched) ---")

    t0 = time.perf_counter()
    _ = search_processpool_good(
        small_queries,
        small_corpus,
        k=5,
        max_workers=4
    )
    t1 = time.perf_counter()

    print(f"Good approach: {t1 - t0:.3f}s (2 queries, 500 documents)")

    # Sequential NumPy baseline
    print("\n--- Sequential NumPy ---")

    benchmark_results = {}

    benchmark_result = benchmark(
        search_numpy_serial,
        queries,
        corpus,
        k=5,
        warmup=3,
        runs=5
    )

    reference = benchmark_result["results"]

    benchmark_results["Sequential NumPy"] = {
        "workers": 1,
        "mean": benchmark_result["mean"],
        "std": benchmark_result["std"],
        "min": benchmark_result["min"],
        "max": benchmark_result["max"]
    }

    print(
        f"Mean: {benchmark_result['mean']:.6f}s, "
        f"Std: {benchmark_result['std']:.6f}s"
    )

    # Benchmark ThreadPool and ProcessPool configurations
    for workers in [1, 2, 4, 8]:
        print(f"\n--- ThreadPool {workers} workers ---")

        benchmark_result = benchmark(
            search_threadpool_numpy,
            queries,
            corpus,
            k=5,
            max_workers=workers,
            warmup=3,
            runs=5
        )

        result = benchmark_result["results"]
        is_correct = verify_results(reference, result)

        benchmark_results[f"ThreadPool {workers}"] = {
            "workers": workers,
            "mean": benchmark_result["mean"],
            "std": benchmark_result["std"],
            "min": benchmark_result["min"],
            "max": benchmark_result["max"]
        }

        print(
            f"Mean: {benchmark_result['mean']:.6f}s, "
            f"Std: {benchmark_result['std']:.6f}s, "
            f"Correctness: {'PASS' if is_correct else 'FAIL'}"
        )

        if not is_correct:
            raise RuntimeError(f"ThreadPool {workers} returned incorrect results.")

    for workers in [1, 2, 4, 8]:
        print(f"\n--- ProcessPool {workers} workers ---")

        benchmark_result = benchmark(
            search_processpool_numpy,
            queries,
            corpus,
            k=5,
            max_workers=workers,
            warmup=3,
            runs=5
        )

        result = benchmark_result["results"]
        is_correct = verify_results(reference, result)

        benchmark_results[f"ProcessPool {workers}"] = {
            "workers": workers,
            "mean": benchmark_result["mean"],
            "std": benchmark_result["std"],
            "min": benchmark_result["min"],
            "max": benchmark_result["max"]
        }

        print(
            f"Mean: {benchmark_result['mean']:.6f}s, "
            f"Std: {benchmark_result['std']:.6f}s, "
            f"Correctness: {'PASS' if is_correct else 'FAIL'}"
        )

        if not is_correct:
            raise RuntimeError(f"ProcessPool {workers} returned incorrect results.")

    # Print comparison table
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    print(
        f"{'Method':<22} {'Mean (s)':>12} "
        f"{'Std (s)':>12} {'Speedup':>12}"
    )

    baseline_time = benchmark_results["Sequential NumPy"]["mean"]

    for name, data in benchmark_results.items():
        speedup = baseline_time / data["mean"]

        print(
            f"{name:<22} {data['mean']:>12.6f} "
            f"{data['std']:>12.6f} {speedup:>11.2f}x"
        )

    # Save CSV and scaling graph
    csv_path = RESULTS_DIR / "processpool_benchmark.csv"
    graph_path = RESULTS_DIR / "processpool_scaling.png"

    save_benchmark_csv(benchmark_results, csv_path)
    plot_scaling(benchmark_results, graph_path)

    print("\nResults saved:")
    print(csv_path)
    print(graph_path)

    print("\n" + "=" * 60)
    print("All benchmark configurations completed.")
    print("=" * 60)
