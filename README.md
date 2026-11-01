# High-Performance Similarity Search Engine

A high-performance semantic similarity search engine developed as part of the YZM 513 project. The project explores different optimization techniques for similarity search, including profiling, matrix operations, dimensionality reduction, vectorization, and parallel computing.

The final system uses precomputed sentence embeddings and NumPy-based vectorized operations to retrieve the top-k most similar documents.

## Project Structure

```text
high-performance-similarity-search/
├── README.md
├── requirements.txt
├── requirements-dataset.txt
├── data/                              # Synthetic document dataset
├── embeddings/                        # Precomputed document and query embeddings
│   ├── 1k/
│   ├── 3k/
│   ├── 6k/
│   └── 12k/
├── results/                           # Experimental outputs and benchmark results
│
├── notebooks/
│   ├── 269797012_Similarity_Search_Dataset_Preparation.ipynb
│   └── report_tables.ipynb
│
├── phase1_baseline/
│   ├── naive_search.py
│   └── profiling_template.py
│
├── phase2_matrix_ops/
│   ├── sparse_dense_compare.py
│   ├── svd_power_iteration.py
│   ├── vectorise_vs_loop.py
│   └── vectorise_vs_loop.ipynb
│
├── phase3_parallelism/
│   ├── async_search.py
│   ├── process_pool_search.py
│   ├── ray_search.py
│   └── thread_pool_search.py
│
├── phase4_final/
│   └── phase4_final_system.ipynb
│
├── benchmark_utils.py
└── prepare_embeddings.py
```

## Project Overview

The project investigates how different computational techniques affect the performance of semantic similarity search as the document corpus grows.

The experiments are organized into four phases:

- **Phase 1 – Baseline and Profiling:** Implementation of a pure Python cosine similarity search algorithm and performance profiling using cProfile, line_profiler, and py-spy.
- **Phase 2 – Matrix Operations:** Comparison of sparse and dense representations, SVD-based dimensionality reduction, and loop-based versus vectorized computations.
- **Phase 3 – Parallelism:** Evaluation of asynchronous execution, thread-based parallelism, and process-based parallelism.
- **Phase 4 – Final System:** Implementation and evaluation of the final vectorized similarity search pipeline, including correctness checks, scalability benchmarks, and an interactive demonstration.

## Dataset and Embeddings

The project uses a synthetic English document dataset containing 12,000 documents.

| Configuration       | Value                                    |
| ------------------- | ---------------------------------------- |
| Dataset             | `synthetic_documents_12000_1b_v2.csv`    |
| Embedding model     | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384                                      |
| Corpus sizes        | 1K, 3K, 6K, 12K                          |
| Number of queries   | 50                                       |
| Query generation    | First 25 words of selected documents     |

The document and query embeddings are stored as NumPy arrays. Document IDs and query information are stored separately as CSV metadata files.

**The dataset and precomputed embeddings are already available in the project directories.** They do not need to be regenerated for normal benchmark execution.

## Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

For dataset generation and preparation dependencies:

```powershell
python -m pip install -r requirements-dataset.txt
```

### 2. Dataset and Embedding Preparation

The dataset preparation notebook is available at:

```text
notebooks/269797012_Similarity_Search_Dataset_Preparation.ipynb
```

To generate document and query embeddings using the existing dataset, run:

```powershell
python prepare_embeddings.py
```

This script:

1. Loads and cleans the document dataset.
2. Selects a reproducible subset of 12,000 documents.
3. Generates 50 queries using the first 25 words of selected documents.
4. Encodes documents and queries using Sentence Transformers.
5. Saves embeddings and metadata for the 1K, 3K, 6K, and 12K corpus sizes.

Generated files are stored under `embeddings/`.

### 3. Running the Experiments

Run the scripts from the project root directory.

#### Phase 1: Baseline and Profiling

Run the naive similarity search:

```powershell
python phase1_baseline/naive_search.py
```

The script loads the 12K corpus and query embeddings, calculates cosine similarity using pure Python loops, and retrieves the top-5 most similar documents for each query.

Run profiling:

```powershell
python phase1_baseline/profiling_template.py
```

The profiling template supports:

- cProfile for function-level performance analysis.
- line_profiler for line-by-line profiling.
- py-spy for flame graph generation.

#### Phase 2: Matrix Operations

Run the sparse and dense representation comparison:

```powershell
python phase2_matrix_ops/sparse_dense_compare.py
```

Run the SVD and power iteration experiments:

```powershell
python phase2_matrix_ops/svd_power_iteration.py
```

Run the vectorization comparison:

```powershell
python phase2_matrix_ops/vectorise_vs_loop.py
```

The corresponding Jupyter notebook is also available:

```text
phase2_matrix_ops/vectorise_vs_loop.ipynb
```

#### Phase 3: Parallelism

Run the thread-based search experiment:

```powershell
python phase3_parallelism/thread_pool_search.py
```

Run the process-based search experiment:

```powershell
python phase3_parallelism/process_pool_search.py
```

Run the asynchronous search experiment:

```powershell
python phase3_parallelism/async_search.py
```

The `ray_search.py` file is retained in the project directory but is not part of the executed experiments or final system.

#### Phase 4: Final System

The final implementation and live demonstration are provided as a Jupyter notebook:

```text
phase4_final/phase4_final_system.ipynb
```

Open and execute the notebook to reproduce the final system experiments.

The final pipeline uses:

- Precomputed document and query embeddings.
- L2 normalization.
- NumPy matrix multiplication for cosine similarity.
- Partial top-k selection using `argpartition`.
- Sorting of the selected top-k results.
- Correctness verification against reference results.

The notebook also includes corpus scaling benchmarks and an interactive query demonstration.

## Benchmarking

The project uses a shared benchmarking utility in `benchmark_utils.py`.

The utility provides:

- Warm-up and repeated execution measurements.
- Mean, standard deviation, minimum, and maximum execution times.
- Peak memory measurement.
- Result verification.
- Benchmark plotting and visualization.

For consistent comparisons:

1. Use the same query embeddings across corpus sizes.
2. Use the same embedding model and vector dimensions.
3. Compare execution times using the same hardware and environment.
4. Report mean execution time and standard deviation.
5. Verify optimized results against the baseline.
6. Store experimental outputs under the appropriate `results/` directories.

## Results and Reproducibility

Experimental outputs, including benchmark measurements, profiling reports, CSV files, and generated figures, are organized under:

```text
results/
├── phase1/
├── phase2/
└── phase3/
└── phase4/
```

Phase 4 benchmark results and visualizations are generated through the final system notebook.

The `notebooks/report_tables.ipynb` notebook contains the tables and visualizations prepared for the project report.

## Requirements

Main project dependencies are listed in `requirements.txt`.

Dataset generation and preparation dependencies are listed separately in `requirements-dataset.txt`.

The project primarily uses:

- Python
- NumPy
- SciPy
- Pandas
- Scikit-learn
- Matplotlib
- Sentence Transformers
- PyTorch
- Numba
- Profiling tools

## Common Issues

- **Missing embeddings:** Ensure the required corpus-size directory exists under `embeddings/`.
- **Incorrect corpus size:** Check that the selected corpus size matches the loaded embedding directory.
- **Missing dataset:** Verify that the CSV file exists under `data/` before running `prepare_embeddings.py`.
- **Missing dependencies:** Install the required packages using the appropriate requirements file.
- **Slow baseline execution:** The Phase 1 implementation uses pure Python loops and is intentionally computationally expensive.
- **Memory issues:** Start with a smaller corpus when testing experimental implementations.
