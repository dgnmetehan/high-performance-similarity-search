"""
Phase 2A: Sparse vs Dense -- The Representation Decision

Before you search fast, you need to decide: how do you store your data?
Sparse (TF-IDF, CSR) or dense (embeddings)?

This is a design decision with real performance consequences.
Your report should justify which representation you chose for your final system.
"""

import numpy as np
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from benchmark_utils import benchmark

def memory_usage(array_or_matrix):
    """Return memory usage in MB."""
    if isinstance(array_or_matrix, csr_matrix):
        return (
            array_or_matrix.data.nbytes
            + array_or_matrix.indices.nbytes
            + array_or_matrix.indptr.nbytes
        ) / (1024 * 1024)
    else:
        # Dense array
        return array_or_matrix.nbytes / (1024 * 1024)



def sparse_dot_sparse(A_sparse, B_sparse):
    """Compute A @ B.T using SciPy sparse operations."""
    #using scipy.sparse
    A = A_sparse.tocsr()
    B = B_sparse.tocsr()

    return (A @ B.T).toarray()

def sparse_dot_sparse_manual(A_sparse, B_sparse):
    """Compute A @ B.T for two sparse CSR matrices."""
    A = A_sparse.tocsr()
    B = B_sparse.tocsr()

    result = np.zeros(
        (A.shape[0], B.shape[0]),
        dtype=np.float64
    )

    # Build feature postings for matrix B
    postings = {}
    for j in range(B.shape[0]):
        for idx in range(B.indptr[j], B.indptr[j + 1]):
            feature = B.indices[idx]
            value = B.data[idx]
            postings.setdefault(feature, []).append((j, value))

    # Accumulate products for matching features
    for i in range(A.shape[0]):
        for idx in range(A.indptr[i], A.indptr[i + 1]):
            feature = A.indices[idx]
            value_a = A.data[idx]

            for j, value_b in postings.get(feature, []):
                result[i, j] += value_a * value_b

    return result




def sparse_dot_dense(A_sparse, B_dense):
    """Compute A @ B.T where A is sparse CSR and B is dense."""
    A = A_sparse.tocsr()

    result = np.zeros(
        (A.shape[0], B_dense.shape[0]),
        dtype=np.float64
    )

    for i in range(A.shape[0]):
        for idx in range(A.indptr[i], A.indptr[i + 1]):
            feature = A.indices[idx]
            value = A.data[idx]
            result[i, :] += value * B_dense[:, feature]

    return result




def dense_dot_dense(A_dense, B_dense):
    """Compute A @ B.T using NumPy."""
    return np.asarray(A_dense) @ np.asarray(B_dense).T



def compare_representations(corpus_texts, query_texts, vectorizer, svd):
    """
    Compare sparse and dense representations at different corpus sizes.

    Returns:
        list of dicts with corpus_size, sparse_mem, dense_mem,
        sparse_sparse_time, sparse_dense_time, dense_dense_time
    """
    corpus_sizes = [1000, 3000, 6000, 12000]
    results = []

    for size in corpus_sizes:
        if size > len(corpus_texts):
            break

        subset = corpus_texts[:size]
        print(f"\n--- Corpus size: {size} ---")

        # Build sparse matrix
        X_sparse = vectorizer.transform(subset)
        X_dense_tfidf = X_sparse.toarray()
        print(f"Sparse shape: {X_sparse.shape}, nnz: {X_sparse.nnz}, "
              f"density: {X_sparse.nnz / (X_sparse.shape[0] * X_sparse.shape[1]):.4f}")

        # Build dense matrix
        X_dense = svd.transform(X_sparse)
        print(f"Dense shape: {X_dense.shape}")

        # Memory comparison
        sparse_mem = memory_usage(X_sparse)
        dense_mem = memory_usage(X_dense_tfidf)
        reduced_mem = memory_usage(X_dense)

        print(
            f"Memory: sparse={sparse_mem:.2f}MB, "
            f"dense TF-IDF={dense_mem:.2f}MB, "
            f"dense SVD={reduced_mem:.2f}MB"
)

        # Query representations
        Q_sparse = vectorizer.transform(query_texts)
        Q_dense_tfidf = Q_sparse.toarray()

        # TODO: benchmark dot products for each combination
        # Record times, store in results

        operations = {
            "sparse_sparse_scipy_time": lambda: sparse_dot_sparse(
                Q_sparse, X_sparse
            ),
            "sparse_sparse_manual_time": lambda: sparse_dot_sparse_manual(
                Q_sparse, X_sparse
            ),
            "sparse_dense_time": lambda: sparse_dot_dense(
                Q_sparse, X_dense_tfidf
            ),
            "dense_dense_time": lambda: dense_dot_dense(
                Q_dense_tfidf, X_dense_tfidf
            ),
        }

        timings = {}
        benchmark_details = {}

        for name, operation in operations.items():
            bench = benchmark(
                operation,
                warmup=3,
                runs=5,
            )

            # Mean time is used in the compute-time plot.
            timings[name] = bench["mean"]

            # Save summary statistics and raw measurements.
            benchmark_details[name] = {
                "mean": bench["mean"],
                "std": bench["std"],
                "min": bench["min"],
                "max": bench["max"],
                "raw_times": bench["raw_times"],
            }

        print(
            "Dot-product mean times: "
            f"sparse-sparse (SciPy)={timings['sparse_sparse_scipy_time']:.4f}s, "
            f"sparse-sparse (Manual)={timings['sparse_sparse_manual_time']:.4f}s, "
            f"sparse-dense={timings['sparse_dense_time']:.4f}s, "
            f"dense-dense={timings['dense_dense_time']:.4f}s"
        )

        result_row = {
            "corpus_size": size,
            "sparse_mem_mb": sparse_mem,
            "dense_mem_mb": dense_mem,
            "reduced_mem_mb": reduced_mem,
            **timings,
        }

        # Add benchmark statistics and individual run times to the CSV data.
        for name, details in benchmark_details.items():
            result_row[f"{name}_std_seconds"] = details["std"]
            result_row[f"{name}_min_seconds"] = details["min"]
            result_row[f"{name}_max_seconds"] = details["max"]
            result_row[f"{name}_raw_times_seconds"] = ";".join(
                f"{t:.6f}" for t in details["raw_times"]
            )

        results.append(result_row)

    return results


def plot_sparse_dense(results, output_path):
    """
    Plot memory and compute comparison.
    TODO: create 2 subplots: memory vs corpus size, time vs corpus size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sizes = [r["corpus_size"] for r in results]

    # Memory plot
    sparse_mems = [r["sparse_mem_mb"] for r in results]
    dense_mems = [r["dense_mem_mb"] for r in results]
    reduced_mems = [r["reduced_mem_mb"] for r in results]

    axes[0].plot(
        sizes, sparse_mems,
        'o-', label='Sparse (CSR)', linewidth=2
    )
    axes[0].plot(
        sizes, dense_mems,
        's-', label='Dense TF-IDF', linewidth=2
    )
    axes[0].plot(
        sizes, reduced_mems,
        '^-', label='Dense SVD (128 components)', linewidth=2
    )
    axes[0].set_xlabel('Corpus Size (documents)')
    axes[0].set_ylabel('Memory (MB)')
    axes[0].set_title('Memory Footprint')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # TODO: add compute time comparison subplot
    # Compute time plot
    axes[1].plot(
        sizes,
        [r["sparse_sparse_scipy_time"] for r in results],
        'o-', label='Sparse-Sparse (SciPy)', linewidth=2
    )
    axes[1].plot(
        sizes,
        [r["sparse_sparse_manual_time"] for r in results],
        's-', label='Sparse-Sparse (Manual)', linewidth=2
    )
    axes[1].plot(
        sizes,
        [r["sparse_dense_time"] for r in results],
        '^-', label='Sparse-Dense', linewidth=2
    )
    axes[1].plot(
        sizes,
        [r["dense_dense_time"] for r in results],
        'd-', label='Dense-Dense', linewidth=2
    )

    axes[1].set_xlabel('Corpus Size (documents)')
    axes[1].set_ylabel('Mean Dot-Product Time (seconds)')    
    axes[1].set_title('Compute Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2A: Sparse vs Dense Comparison")
    print("=" * 60)


    project_dir = Path(__file__).resolve().parent.parent
    data_path = project_dir / "data" / "synthetic_documents_12000_1b_v2.csv"

    df = pd.read_csv(data_path)
    corpus_texts = df["text"].fillna("").astype(str).tolist()

    if len(corpus_texts) < 12000:
        raise ValueError(
            f"Expected 12000 documents, found {len(corpus_texts)}"
        )

    # Use 100 documents as queries for benchmarking
    query_texts = corpus_texts[-100:]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    X_full = vectorizer.fit_transform(corpus_texts)

    svd = TruncatedSVD(n_components=128, random_state=42)
    svd.fit(X_full)

    output_dir = project_dir / "results" / "phase2" / "2a"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = compare_representations(
        corpus_texts, query_texts, vectorizer, svd
    )

    plot_path = output_dir / "sparse_dense_comparison.png"
    plot_sparse_dense(results, plot_path)

    results_df = pd.DataFrame(results)
    csv_path = output_dir / "sparse_dense_results.csv"
    results_df.to_csv(csv_path, index=False)

    print(f"Saved measurements: {csv_path}")

