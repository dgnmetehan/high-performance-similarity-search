"""
Phase 2B: Dimensionality Reduction -- Choose and Justify

Your corpus might have 5K-20K documents with 768-dim embeddings.
That's a lot of numbers to search through. Dimensionality reduction
is your first real speedup -- but which method should you choose?

This is a design decision. Your report must justify your choice with data.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD, PCA, IncrementalPCA

import csv
from pathlib import Path

import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from benchmark_utils import benchmark, plot_scaling




OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "phase2" / "2b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def power_iteration_svd(X, k, max_iter=100, tol=1e-6):
    """
    Compute top-k singular vectors using power iteration.

    This is YOUR implementation. The core algorithm:
    1. Start with random orthonormal V (n x k)
    2. Iterate: V_new = qr(X^T @ X @ V)
    3. Stop when converged
    4. Compute U and S from XV

    Args:
        X: (m x n) matrix
        k: number of components to keep
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        U_k (m x k), S_k (k,), V_k^T (k x n)
    """
    m, n = X.shape

    if not 1 <= k <= min(m, n):
        raise ValueError("k must be between 1 and min(X.shape)")

    # Compute the Gram matrix
    gram = X.T @ X

    # Initialize an orthonormal basis
    rng = np.random.default_rng(42)
    V, _ = np.linalg.qr(rng.standard_normal((n, k)))

    # Power iteration
    for _ in range(max_iter):
        V_new, _ = np.linalg.qr(gram @ V)

        # Compare subspaces, avoiding sign ambiguity.
        delta = np.linalg.norm(
            V_new @ V_new.T - V @ V.T,
            ord="fro",
        )

        V = V_new
        if delta < tol:
            break

    # Orthonormal basis for the column space of X @ V.
    # Compute final U, S, and Vt
    Y = X @ V
    U, S, Wt = np.linalg.svd(Y, full_matrices=False)
    Vt = Wt @ V.T

    return U, S, Vt


def reconstruction_error(X, U, S, Vt):
    """
    Compute ||X - U @ diag(S) @ Vt||_F (Frobenius norm).
    Lower is better.
    """
    X_recon = U @ np.diag(S) @ Vt
    return np.linalg.norm(X - X_recon)


def compare_svd_methods(X, k_values):
    """ 
    Compare your power iteration against library implementations.

    Methods to compare (choose at least 2 library methods):
    - Your power_iteration_svd  
    - np.linalg.svd (full SVD)
    - sklearn TruncatedSVD
    - sklearn PCA
    - sklearn IncrementalPCA

    Returns:dict mapping method_name -> list of (k, mean_time, std_time, error) tuples
    """
    results = {
        "Power Iteration": [],
        "TruncatedSVD": [],
        "PCA": [],
    }

    for k in k_values:
        if k > min(X.shape):
            continue

        print(f"\nBenchmarking k={k}...")

        # 1. Custom Power Iteration
        bench = benchmark(
            power_iteration_svd,
            X,
            k,
            warmup=3,
            runs=5,
        )
        U, S, Vt = bench["results"]
        error = reconstruction_error(X, U, S, Vt)

        results["Power Iteration"].append(
            (k, bench["mean"], bench["std"], error, bench["raw_times"])
        )
        print(
            f"Power Iteration: {bench['mean']:.4f}s "
            f"(std={bench['std']:.4f}), error={error:.2f}"
        )   

        # 2. TruncatedSVD
        def run_svd():
            model = TruncatedSVD(
                n_components=k,
                algorithm="randomized",
                n_iter=7,
                random_state=42,
            )
            Z = model.fit_transform(X)
            return model, Z

        bench = benchmark(run_svd, warmup=3, runs=5)
        model, Z = bench["results"]
        error = float(np.linalg.norm(
            X - model.inverse_transform(Z), ord="fro"
        ))

        results["TruncatedSVD"].append(
            (k, bench["mean"], bench["std"], error, bench["raw_times"])
        )
        print(
            f"TruncatedSVD: {bench['mean']:.4f}s "
            f"(std={bench['std']:.4f}), error={error:.2f}"
        )

        # 3. PCA
        def run_pca():
            model = PCA(
                n_components=k,
                svd_solver="randomized",
                random_state=42,
            )
            Z = model.fit_transform(X)
            return model, Z

        bench = benchmark(run_pca, warmup=3, runs=5)
        model, Z = bench["results"]
        error = float(np.linalg.norm(
            X - model.inverse_transform(Z), ord="fro"
        ))
        results["PCA"].append(
            (k, bench["mean"], bench["std"], error, bench["raw_times"])
        )
        print(
            f"PCA: {bench['mean']:.4f}s "
            f"(std={bench['std']:.4f}), error={error:.2f}"
        )

    return results



def save_comparison_csv(results, filepath):
    """Save summary statistics and raw timing measurements."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "k",
            "mean_time_seconds",
            "std_time_seconds",
            "reconstruction_error",
            "raw_times_seconds",
        ])

        for method, measurements in results.items():
            for k, mean_time, std_time, error, raw_times in measurements:
                writer.writerow([
                    method,
                    k,
                    mean_time,
                    std_time,
                    error,
                    ";".join(f"{t:.6f}" for t in raw_times),
                ])

    print(f"Saved measurements: {filepath}")



def plot_svd_comparison(results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    markers = {
    "Power Iteration": "o",
    "TruncatedSVD": "s",
    "PCA": "^",
    }

    for method_name, values in results.items():
        marker = markers.get(method_name, "o")
        k_values = [item[0] for item in values]
        times = [item[1] for item in values]
        stds = [item[2] for item in values]
        errors = [item[3] for item in values]

        axes[0].errorbar(   
            k_values, times, yerr=stds,
            marker=marker, capsize=4, label=method_name
        )
        axes[1].plot(
            k_values, errors,
            marker=marker, label=method_name
        )

    axes[0].set_xlabel("Number of Components (k)")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Execution Time vs. k")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Number of Components (k)")
    axes[1].set_ylabel("Reconstruction Error")
    axes[1].set_title("Reconstruction Error vs. k")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_scaling_comparison(matrix_sizes, k=50):
    """Compare method runtimes across different matrix sizes."""
    methods = {
        "Power Iteration": [],
        "TruncatedSVD": [],
        "PCA": [],
    }
    row_sizes = []

    for m, n in matrix_sizes:
        if k > min(m, n):
            continue

        print(f"\nScaling test: {m} x {n}")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((m, n)).astype(np.float64)
        row_sizes.append(m)

        bench = benchmark(
            power_iteration_svd,
            X, k,
            max_iter=100,
            warmup=3,
            runs=5,
        )
        methods["Power Iteration"].append(bench["mean"])
        print(f"Power Iteration: {bench['mean']:.4f}s")

        def run_svd():
            model = TruncatedSVD(
                n_components=k,
                algorithm="randomized",
                n_iter=7,
                random_state=42,
            )
            return model.fit_transform(X)

        bench = benchmark(run_svd, warmup=3, runs=5)
        methods["TruncatedSVD"].append(bench["mean"])
        print(f"TruncatedSVD: {bench['mean']:.4f}s")

        def run_pca():
            model = PCA(
                n_components=k,
                svd_solver="randomized",
                random_state=42,
            )
            return model.fit_transform(X)

        bench = benchmark(run_pca, warmup=1, runs=3)
        methods["PCA"].append(bench["mean"])
        print(f"PCA: {bench['mean']:.4f}s")

    plot_scaling(
        x_values=row_sizes,
        y_values_dict=methods,
        xlabel="Matrix Rows (m)",
        ylabel="Mean Execution Time (seconds)",
        title=f"Scaling with Matrix Size (k={k})",
        save_path=str(OUTPUT_DIR / "svd_scaling.png"),
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2B: Dimensionality Reduction")
    print("=" * 60)
    
    # Generate test data 
    rng = np.random.default_rng(42)
    m, n = 5000, 768
    X = rng.standard_normal((m, n)).astype(np.float64)
    print(f"Test matrix: {m} x {n}")

    # First: verify your power iteration is correct

    print("\nVerifying Power Iteration SVD...")
    k_test = 50
    U_pi, S_pi, Vt_pi = power_iteration_svd(X, k_test)
    print(f"Shapes: U={U_pi.shape}, S={S_pi.shape}, Vt={Vt_pi.shape}")

    # Compare against numpy (ground truth)
    U_np, S_np, Vt_np = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    err_yours = reconstruction_error(X, U_pi, S_pi, Vt_pi)
    err_numpy = reconstruction_error(
        X,
        U_np[:, :k_test],
        S_np[:k_test],
        Vt_np[:k_test, :],
    )
    print(
        f"Reconstruction error -- yours: {err_yours:.2f}, "
        f"NumPy rank-{k_test}: {err_numpy:.2f}"
    )
    # Benchmark across k values
    k_values = [10, 25, 50, 100, 200]
    results = compare_svd_methods(X, k_values)

    plot_svd_comparison(
        results,
        OUTPUT_DIR / "svd_comparison.png",
    )
    save_comparison_csv(
        results,
        OUTPUT_DIR / "svd_comparison_results.csv",
    )

    matrix_sizes = [
        (1000, 768),
        (2500, 768),
        (5000, 768),
        (10000, 768),
    ]
    plot_scaling_comparison(matrix_sizes, k=50)

    print("\n" + "=" * 60)
    print("\nPhase 2B complete.")

