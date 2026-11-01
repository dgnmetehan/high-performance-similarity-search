
"""
Phase 1: Profiling Template

Profile the naive search using cProfile and line_profiler.
"""

import cProfile
import pstats
import io
from pathlib import Path

import numpy as np

# Import your naive search
from naive_search import search_naive, cosine_similarity_python

# --------------------------------------------------
# 1. cProfile
# --------------------------------------------------
def profile_with_cprofile(queries, corpus, k=5, output_file=None):
    """
    Profile the search function with cProfile.
    Shows which functions take the most cumulative time.
    """

    print("=" * 60)
    print("cProfile Output (sorted by cumulative time)")
    print("=" * 60)

    profiler = cProfile.Profile()
    profiler.enable()
    results = search_naive(queries, corpus, k)
    profiler.disable()

    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output)
    stats.sort_stats("cumtime")
    stats.print_stats(20)

    report = output.getvalue()
    print(report)

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report, encoding="utf-8")
        print(f"cProfile report saved to: {output_file}")

    return results


# --------------------------------------------------
# 2. Line profiler
# Run separately with kernprof when needed.
# --------------------------------------------------

def profile_with_line_profiler(
    queries,
    corpus,
):
    """
    Profile line-by-line with line_profiler.
    Shows which lines inside cosine_similarity_python are slowest.

    Requires: pip install line_profiler
    Usage: kernprof -l -v profiling_template.py
    """

    print("=" * 60)
    print("Line Profiler")
    print("=" * 60)

    # Decorate the target function
    profiled_cosine = profile(cosine_similarity_python)


    # Run a small subset (line_profiler is slow)
    small_queries = queries[:10]
    small_corpus = corpus[:1000]

    for query in small_queries:
        for doc in small_corpus:
            profiled_cosine(query, doc)



            

def profile_with_pyspy():
    """
    Generate a flame graph with py-spy.

    Run from terminal:
        py-spy record -o flamegraph.svg -- python naive_search.py

    This gives you a visual flame graph showing where time is spent.
    """
    print("Run this from the project root:")
    print(
        r"py-spy record -o .\results\phase1\12k\flamegraph.svg "
        r"-- python .\phase1_baseline\naive_search.py"
    )

if __name__ == "__main__":
    # Load data
    project_dir = Path(__file__).resolve().parent.parent
    embeddings_dir = project_dir / "embeddings" / "12k"
    results_dir = project_dir / "results" / "phase1" / "12k"

    try:
        corpus = np.load(
            embeddings_dir / "corpus_embeddings.npy"
        ).tolist()
        queries = np.load(
            embeddings_dir / "query_embeddings.npy"
        ).tolist()
    except FileNotFoundError as exc:
        print(f"Embedding file not found: {exc.filename}")
        raise SystemExit(1)

    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)} queries")
    print(f"Dimensions: {len(corpus[0])}")

    # # 1. Run cProfile first.
    profile_with_cprofile(
        queries,
        corpus,
        k=5,
        output_file=results_dir / "phase1_cprofile.txt",
    )

    # use this command to run:    kkernprof -l -v -o .\results\phase1\12k\phase1_line_profile.lprof .\phase1_baseline\profiling_template.py
    # 2. Uncomment this call when running line_profiler with kernprof.

    # profile_with_line_profiler(queries, corpus)

    print("\n" + "=" * 60)
    profile_with_pyspy()
    