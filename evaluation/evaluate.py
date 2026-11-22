import sys
import json
import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVALUATION_DIR / "evaluation_questions.json"
RESULTS_FILE = EVALUATION_DIR / "evaluation_results.json"
REPORT_FILE = EVALUATION_DIR / "evaluation_report.csv"

sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import (
    load_embedding_model,
    load_chunks,
    load_faiss_index,
    search_policy,
    TOP_K
)

from src.llm import generate_response

def save_results(results):

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

def prepare_retrieved_chunks(retrieved_chunks):

    formatted_chunks = []

    for chunk in retrieved_chunks:

        if isinstance(chunk, dict):

            formatted_chunks.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "document_name": chunk.get("document_name", ""),
                "section": chunk.get("section", ""),
                "similarity_score": chunk.get(
                    "similarity_score",
                    chunk.get("score", None)
                ),
                "text": chunk.get("text", "")
            })

        else:

            formatted_chunks.append({
                "retrieved_data": str(chunk)
            })

    return formatted_chunks

def generate_csv_report(results):

    fieldnames = [
        "question_id",
        "question",
        "expected_answer",
        "generated_answer",
        "retrieved_sections",
        "evaluation_status",
        "score"
    ]

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            retrieved_sections = []

            for chunk in result["retrieved_chunks"]:

                if "document_name" in chunk:

                    section = (
                        f'{chunk["document_name"]} - '
                        f'{chunk["section"]}'
                    )

                    retrieved_sections.append(section)

            writer.writerow({

                "question_id": result["question_id"],

                "question": result["question"],

                "expected_answer": result["expected_answer"],

                "generated_answer": result["generated_answer"],

                "retrieved_sections": " | ".join(
                    retrieved_sections
                ),

                # These are intentionally left for manual review
                "evaluation_status": "Not reviewed",

                "score": ""

            })

def main():

    print("\n" + "=" * 65)
    print("       COMPANY POLICY ASSISTANT - EVALUATION")
    print("=" * 65)

    if not QUESTIONS_FILE.exists():

        print("\nERROR: evaluation_questions.json not found.")

        return

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        evaluation_data = json.load(file)

    questions = evaluation_data["questions"]

    print(f"\nTotal Questions: {len(questions)}")

    print("\nLoading embedding model...")

    embedding_model = load_embedding_model()

    print("Loading document chunks...")

    chunks = load_chunks()

    print("Loading FAISS index...")

    index = load_faiss_index()

    print("\nRAG components loaded successfully.")

    results = []

    for position, question_data in enumerate(
        questions,
        start=1
    ):

        question_id = question_data["id"]

        question = question_data["question"]

        expected_answer = question_data["expected_answer"]

        print("\n" + "-" * 65)

        print(
            f"QUESTION {position}/{len(questions)} "
            f"({question_id})"
        )

        print("-" * 65)

        print("\nQuestion:")
        print(question)

        try:
            print("\nRetrieving relevant policy sections...")

            retrieved_chunks = search_policy(
                question,
                embedding_model,
                index,
                chunks,
                top_k=TOP_K
            )

            formatted_chunks = prepare_retrieved_chunks(
                retrieved_chunks
            )

            print(
                f"Retrieved {len(retrieved_chunks)} chunks."
            )

            print("\nGenerating answer using OpenRouter...")

            generated_answer = generate_response(
                question,
                retrieved_chunks
            )

            print("\nGenerated Answer:")
            print(generated_answer)

            result = {

                "question_id": question_id,

                "question": question,

                "expected_answer": expected_answer,

                "expected_sources": question_data.get(
                    "expected_sources",
                    []
                ),

                "key_facts": question_data.get(
                    "key_facts",
                    []
                ),

                "retrieved_chunks": formatted_chunks,

                "generated_answer": generated_answer,

                "evaluation_status": "Not reviewed",

                "score": None

            }

        except Exception as error:

            print("\nERROR while processing question:")
            print(error)

            result = {

                "question_id": question_id,

                "question": question,

                "expected_answer": expected_answer,

                "expected_sources": question_data.get(
                    "expected_sources",
                    []
                ),

                "key_facts": question_data.get(
                    "key_facts",
                    []
                ),

                "retrieved_chunks": [],

                "generated_answer": None,

                "evaluation_status": "Execution Error",

                "score": None,

                "error": str(error)

            }

        results.append(result)

        # Save after every question to preserve completed results
        save_results(results)

        print("\nQuestion processing completed.")

    print("\n" + "=" * 65)

    print("GENERATING EVALUATION REPORT")

    generate_csv_report(results)

    print("\nEvaluation completed.")

    print("\nFiles generated:")

    print(f"1. {RESULTS_FILE}")

    print(f"2. {REPORT_FILE}")

    successful = sum(
        1 for result in results
        if result["generated_answer"] is not None
    )

    failed = len(results) - successful

    print("\n" + "=" * 65)

    print("EVALUATION SUMMARY")

    print("=" * 65)

    print(f"Total Questions : {len(questions)}")

    print(f"Successful      : {successful}")

    print(f"Failed          : {failed}")

    print("\nCorrectness scores are pending manual review.")

    print("=" * 65)

if __name__ == "__main__":
    main()