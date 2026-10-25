#!/usr/bin/env python3
"""
Batch evaluation for the NASA Mission Intelligence RAG system.

Loads questions from evaluation_dataset.txt and runs each question through:

1. ChromaDB retrieval
2. Context construction
3. LLM response generation
4. RAGAS evaluation
5. Aggregate metric calculation
"""

import json
import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

import rag_client
import llm_client
import ragas_evaluator


# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------

load_dotenv()


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

EVALUATION_FILE = "evaluation_dataset.txt"

CHROMA_DIR = "./chroma_db"

COLLECTION_NAME = "nasa_space_missions_text"

MODEL = "gpt-4o-mini"

N_RESULTS = 3

RESULTS_FILE = "evaluation_results.json"


# ------------------------------------------------------------
# Load evaluation questions
# ------------------------------------------------------------

def load_questions(file_path: str) -> List[str]:
    """Load evaluation questions from a text file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {file_path}"
        )

    questions = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
        and not line.strip().startswith("#")
    ]

    if not questions:
        raise ValueError(
            "Evaluation dataset contains no questions."
        )

    return questions


# ------------------------------------------------------------
# Evaluate one question
# ------------------------------------------------------------

def evaluate_question(
    collection,
    question: str,
    openai_key: str
) -> Dict:
    """Run one question through the complete RAG pipeline."""

    if not question.strip():
        return {
            "question": question,
            "error": "Question is empty."
        }

    # --------------------------------------------------------
    # Retrieve relevant NASA documents
    # --------------------------------------------------------

    retrieval_result = (
        rag_client.retrieve_documents(
            collection=collection,
            query=question,
            n_results=N_RESULTS
        )
    )

    if (
        not retrieval_result
        or not retrieval_result.get("documents")
        or not retrieval_result["documents"][0]
    ):
        return {
            "question": question,
            "error": "No documents were retrieved."
        }

    documents = retrieval_result[
        "documents"
    ][0]

    metadatas = (
        retrieval_result.get(
            "metadatas"
        )
        or [[]]
    )[0]

    # --------------------------------------------------------
    # Construct LLM context
    # --------------------------------------------------------

    context = rag_client.format_context(
        documents,
        metadatas
    )

    if not context.strip():
        return {
            "question": question,
            "error": "Retrieved context is empty."
        }

    # --------------------------------------------------------
    # Generate RAG response
    # --------------------------------------------------------

    answer = llm_client.generate_response(
        openai_key=openai_key,
        user_message=question,
        context=context,
        conversation_history=[],
        model=MODEL
    )

    if not answer or not answer.strip():
        return {
            "question": question,
            "error": "The LLM returned an empty answer."
        }

    # --------------------------------------------------------
    # RAGAS evaluation
    # --------------------------------------------------------

    scores = (
        ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=documents
        )
    )

    if "error" in scores:
        return {
            "question": question,
            "answer": answer,
            "error": f"Evaluation failed: {scores['error']}"
        }

    return {
        "question": question,
        "answer": answer,
        "sources": [
            (metadata or {}).get(
                "file_name",
                "unknown"
            )
            for metadata in metadatas
        ],
        "scores": scores
    }


# ------------------------------------------------------------
# Aggregate metrics
# ------------------------------------------------------------

def aggregate_scores(
    results: List[Dict]
) -> Dict[str, float]:
    """Average each metric across successful questions."""

    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for result in results:

        for metric, value in (
            result.get(
                "scores",
                {}
            ).items()
        ):

            if not isinstance(
                value,
                (int, float)
            ):
                continue

            # RAGAS can return NaN when
            # a metric cannot be computed.
            if value != value:
                continue

            totals[metric] = (
                totals.get(
                    metric,
                    0.0
                )
                + float(value)
            )

            counts[metric] = (
                counts.get(
                    metric,
                    0
                )
                + 1
            )

    return {
        metric:
            totals[metric]
            / counts[metric]
        for metric in totals
        if counts.get(metric, 0) > 0
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    openai_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not openai_key:
        raise SystemExit(
            "OPENAI_API_KEY is not configured. "
            "Add it to your .env file."
        )

    # Default to the Vocareum endpoint.
    os.environ.setdefault(
        "OPENAI_BASE_URL",
        "https://openai.vocareum.com/v1"
    )

    questions = load_questions(
        EVALUATION_FILE
    )

    collection = (
        rag_client.initialize_rag_system(
            CHROMA_DIR,
            COLLECTION_NAME
        )
    )

    print(
        f"Loaded {len(questions)} questions. "
        f"Collection has "
        f"{collection.count()} chunks.\n"
    )

    results: List[Dict] = []

    for number, question in enumerate(
        questions,
        start=1
    ):

        print(
            f"[{number}/{len(questions)}] "
            f"{question}"
        )

        # One failing question should
        # not stop the whole batch.
        try:

            result = evaluate_question(
                collection,
                question,
                openai_key
            )

        except Exception as exc:

            result = {
                "question": question,
                "error": str(exc)
            }

        results.append(
            result
        )

        if "error" in result:

            print(
                f"    ERROR: "
                f"{result['error']}\n"
            )

        else:

            for metric, value in (
                result["scores"].items()
            ):

                if isinstance(
                    value,
                    (int, float)
                ):

                    print(
                        f"    {metric}: "
                        f"{value:.3f}"
                    )

                else:

                    print(
                        f"    {metric}: "
                        f"{value}"
                    )

            print()

    averages = aggregate_scores(
        results
    )

    succeeded = sum(
        1
        for result in results
        if "scores" in result
    )

    print(
        "=" * 50
    )

    print(
        f"Evaluated {succeeded} "
        f"of {len(results)} "
        "questions successfully"
    )

    print(
        "Average scores:"
    )

    for metric, value in (
        averages.items()
    ):

        print(
            f"    {metric}: "
            f"{value:.3f}"
        )

    Path(
        RESULTS_FILE
    ).write_text(
        json.dumps(
            {
                "model": MODEL,
                "n_results": N_RESULTS,
                "questions_evaluated":
                    succeeded,
                "questions_total":
                    len(results),
                "average_scores":
                    averages,
                "results":
                    results,
            },
            indent=2
        ),
        encoding="utf-8"
    )

    print(
        f"\nFull results saved to "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()