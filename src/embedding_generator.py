import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOKEN_LIMIT = 512

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHUNKS_FILE = PROJECT_ROOT / "output" / "chunks.json"
EMBEDDINGS_FILE = PROJECT_ROOT / "output" / "embeddings.npz"

def load_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully!")
    print(f"Embedding dimensions: {model.get_embedding_dimension()}")
    print(f"Maximum sequence length: {model.max_seq_length}")

    return model

def load_chunks():

    print("\nLoading chunks...")

    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Total chunks loaded: {len(chunks)}")

    return chunks

def validate_token_counts(model, chunks):

    print("\nValidating token counts...")

    tokenizer = model.tokenizer

    for chunk in chunks:

        tokens = tokenizer.encode(
            chunk["text"],
            add_special_tokens=True
        )

        if len(tokens) > MAX_TOKEN_LIMIT:

            raise ValueError(
                f"Chunk {chunk['chunk_id']} exceeds "
                f"the {MAX_TOKEN_LIMIT}-token limit."
            )

    print("All chunks passed token validation.")

def generate_embeddings(model, chunks):

    print("\nGenerating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype(np.float32)

    print("\nEmbedding generation completed!")

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Data type: {embeddings.dtype}")

    return embeddings

def save_embeddings(chunks, embeddings):

    print("\nSaving embeddings...")

    chunk_ids = np.array(
        [chunk["chunk_id"] for chunk in chunks]
    )

    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings,
        chunk_ids=chunk_ids
    )

    print(f"Embeddings saved to: {EMBEDDINGS_FILE}")

def main():

    model = load_embedding_model()

    chunks = load_chunks()

    validate_token_counts(model, chunks)

    embeddings = generate_embeddings(model, chunks)

    save_embeddings(chunks, embeddings)

    print("\nPipeline completed successfully!")


if __name__ == "__main__":

    main()