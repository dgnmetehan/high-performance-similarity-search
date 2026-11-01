
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "embeddings"

CSV_FILE = DATA_DIR / "synthetic_documents_12000_1b_v2.csv"

# Corpus sizes to prepare
CORPUS_SIZES = [1000, 3000, 6000, 12000]

# Experiment settings
NUMBER_OF_QUERIES = 50
QUERY_WORDS = 25
RANDOM_SEED = 42
BATCH_SIZE = 32

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("Loading document dataset...")
    df = pd.read_csv(CSV_FILE)

    if "text" not in df.columns or "doc_id" not in df.columns:
        raise ValueError(
            "Dataset must contain 'doc_id' and 'text' columns."
        )

    # Clean documents
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset=["doc_id"])

    max_corpus_size = max(CORPUS_SIZES)

    if len(df) < max_corpus_size:
        raise ValueError(
            f"Expected at least {max_corpus_size} valid documents, "
            f"found {len(df)}."
        )

    # Create one reproducible ordering for all corpus sizes.
    # Smaller corpora are subsets of the larger corpora.
    ordered_df = df.sample(
        n=max_corpus_size,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # Select queries from the first 1,000 documents.
    # Their source documents will exist in every corpus size.
    query_df = ordered_df.iloc[:CORPUS_SIZES[0]].sample(
        n=NUMBER_OF_QUERIES,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    query_texts = [
        " ".join(text.split()[:QUERY_WORDS])
        for text in query_df["text"]
    ]

    print(f"Available documents: {len(ordered_df)}")
    print(f"Corpus sizes: {CORPUS_SIZES}")
    print(f"Number of queries: {len(query_texts)}")
    print(f"Embedding model: {MODEL_NAME}")

    # Load the model once.
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    # Encode queries once and reuse them for every corpus size.
    print("\nEncoding queries...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)

    for corpus_size in CORPUS_SIZES:
        print(f"\nPreparing corpus size: {corpus_size}")

        corpus_df = ordered_df.iloc[:corpus_size].copy()

        print(f"Encoding {len(corpus_df)} documents...")
        corpus_embeddings = model.encode(
            corpus_df["text"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype(np.float32)

        # Save each corpus size to a separate directory.
        size_label = f"{corpus_size // 1000}k"
        size_dir = OUTPUT_DIR / size_label
        size_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            size_dir / "corpus_embeddings.npy",
            corpus_embeddings
        )
        np.save(
            size_dir / "query_embeddings.npy",
            query_embeddings
        )

        # Save document IDs in the same order as corpus embeddings.
        corpus_df[["doc_id"]].to_csv(
            size_dir / "corpus_metadata.csv",
            index=False
        )

        # Save query IDs, source document IDs, and query text.
        pd.DataFrame({
            "query_id": [
                f"query_{i + 1:03d}"
                for i in range(len(query_texts))
            ],
            "source_doc_id": query_df["doc_id"].tolist(),
            "query_text": query_texts
        }).to_csv(
            size_dir / "query_metadata.csv",
            index=False
        )

        print(f"Corpus embedding shape: {corpus_embeddings.shape}")
        print(f"Query embedding shape: {query_embeddings.shape}")
        print(f"Saved to: {size_dir}")

    print("\nAll embedding sets prepared successfully.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
