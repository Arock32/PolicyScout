import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
TOP_K = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "output" / "chunks.json"
EMBEDDINGS_FILE = PROJECT_ROOT / "output" / "embeddings.npz"
FAISS_INDEX_FILE = PROJECT_ROOT / "output" / "policy_index.faiss"


# BGE recommended instruction for search queries
QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)

def load_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully!")

    return model

def load_chunks():

    print("Loading original chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Total chunks loaded: {len(chunks)}")

    return chunks

# ============================================================
# ONE-TIME FAISS INDEX CREATION
# Uncomment only when rebuilding the FAISS index.
# Do not execute during normal application usage.
# ============================================================

# def create_faiss_index(embeddings):
#
#     print("\nCreating FAISS index...")
#
#     dimension = embeddings.shape[1]
#
#     index = faiss.IndexFlatIP(dimension)
#
#     print("Index type: IndexFlatIP")
#     print(f"Vector dimensions: {dimension}")
#
#     return index
#
#
# def add_embeddings(index, embeddings):
#
#     print("\nAdding embeddings to FAISS...")
#
#     embeddings = np.ascontiguousarray(
#         embeddings,
#         dtype=np.float32
#     )
#
#     index.add(embeddings)
#
#     print(f"Vectors inserted: {index.ntotal}")
#
#     return index
#
#
# def save_faiss_index(index):
#
#     print("\nSaving FAISS index...")
#
#     faiss.write_index(index, str(FAISS_INDEX_FILE))
#
#     print(f"Index saved to: {FAISS_INDEX_FILE}")

def load_faiss_index():

    print("\nLoading FAISS index...")

    index = faiss.read_index(str(FAISS_INDEX_FILE))

    print("FAISS index loaded successfully!")
    print(f"Total indexed vectors: {index.ntotal}")
    print(f"Vector dimensions: {index.d}")

    return index

def validate_index_mapping(index, chunks):

    print("\nValidating index and chunk mapping...")

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index count does not match the chunk count."
        )

    if index.d != 384:
        raise ValueError(
            f"Expected 384 dimensions, got {index.d}."
        )

    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs detected.")

    print("Index and chunk mapping validated successfully.")

def generate_query_embedding(model, query):

    formatted_query = QUERY_PREFIX + query

    query_embedding = model.encode(
        [formatted_query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.ascontiguousarray(
        query_embedding,
        dtype=np.float32
    )

    return query_embedding

def search_policy(query, model, index, chunks, top_k=TOP_K):

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    if index.ntotal == 0:
        raise ValueError("FAISS index is empty.")

    top_k = min(top_k, index.ntotal)

    print(f"\nSearching for: {query}")

    query_embedding = generate_query_embedding(
        model,
        query
    )

    scores, positions = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, position in zip(scores[0], positions[0]):

        if position < 0:
            continue

        chunk = chunks[int(position)]

        result = {
            "chunk_id": chunk["chunk_id"],
            "document_name": chunk["document_name"],
            "document_type": chunk["document_type"],
            "section": chunk["section"],
            "topic": chunk["topic"],
            "similarity_score": float(score),
            "text": chunk["text"]
        }

        results.append(result)

    return results

def display_results(results):

    print("\n========== SEARCH RESULTS ==========")

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\nResult {rank}")
        print("-" * 50)

        print("Chunk ID:", result["chunk_id"])
        print("Document:", result["document_name"])
        print("Document Type:", result["document_type"])
        print("Section:", result["section"])
        print("Topic:", result["topic"])

        print(
            "Similarity Score:",
            round(result["similarity_score"], 4)
        )

        print("\nOriginal Text:")
        print(result["text"])

def main():

    model = load_embedding_model()

    chunks = load_chunks()

    index = load_faiss_index()

    validate_index_mapping(index, chunks)

    query = "How many hours should an employee work daily?"

    results = search_policy(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=TOP_K
    )

    display_results(results)


if __name__ == "__main__":

    main()