"""
RAGAS evaluation for the NASA Mission Intelligence RAG system.

Core metrics (always computed, no reference needed):
    response_relevancy  - does the answer address the question?
    faithfulness        - is every claim in the answer supported by the
                          retrieved context?

Reference-based metrics (computed only when a reference answer is given):
    bleu_score          - n-gram overlap between answer and reference
    rouge_score         - longest-common-subsequence overlap with reference
    context_precision   - were the retrieved chunks useful for producing
                          the reference answer? (LLM-judged)

Each metric is set up and scored independently, so a missing optional
package or a failure in one metric never blocks the others.
"""

import os
import asyncio
from typing import Any, Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
    )
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    RAGAS_AVAILABLE = True
    RAGAS_IMPORT_ERROR = ""

except ImportError as exc:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(exc)


EVALUATOR_MODEL = "gpt-4o-mini"
EVALUATOR_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BASE_URL = "https://openai.vocareum.com/v1"

CORE_METRICS = ("response_relevancy", "faithfulness")
REFERENCE_METRICS = ("bleu_score", "rouge_score", "context_precision")


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

def _validate_inputs(
    question: Any,
    answer: Any,
    contexts: Any,
    reference: Any,
) -> Optional[str]:
    """Return an error message for invalid input, or None if valid."""

    if not isinstance(question, str) or not question.strip():
        return "question must be a non-empty string"

    if not isinstance(answer, str) or not answer.strip():
        return "answer must be a non-empty string"

    if not isinstance(contexts, list) or not contexts:
        return "contexts must be a non-empty list of strings"

    if not all(isinstance(c, str) and c.strip() for c in contexts):
        return "every item in contexts must be a non-empty string"

    if reference is not None and (
        not isinstance(reference, str) or not reference.strip()
    ):
        return "reference must be a non-empty string or None"

    return None


# ---------------------------------------------------------
# Metric construction
# ---------------------------------------------------------

def _build_metrics(
    api_key: str,
    use_reference: bool,
    metric_errors: Dict[str, str],
) -> Dict[str, Any]:
    """
    Create each metric in its own try block. A metric that cannot be
    created is recorded in metric_errors and left out.
    """

    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

    metrics: Dict[str, Any] = {}

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=EVALUATOR_MODEL,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
    )

    try:
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=EVALUATOR_EMBEDDING_MODEL,
                api_key=api_key,
                base_url=base_url,
            )
        )
        metrics["response_relevancy"] = ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
    except Exception as exc:
        metric_errors["response_relevancy"] = str(exc)

    try:
        metrics["faithfulness"] = Faithfulness(llm=evaluator_llm)
    except Exception as exc:
        metric_errors["faithfulness"] = str(exc)

    if not use_reference:
        return metrics

    # Optional metrics: need extra packages (sacrebleu, rouge-score,
    # rapidfuzz) and a real reference answer.
    try:
        from ragas.metrics import BleuScore
        metrics["bleu_score"] = BleuScore()
    except Exception as exc:
        metric_errors["bleu_score"] = str(exc)

    try:
        from ragas.metrics import RougeScore
        metrics["rouge_score"] = RougeScore()
    except Exception as exc:
        metric_errors["rouge_score"] = str(exc)

    try:
        from ragas.metrics import LLMContextPrecisionWithReference
        metrics["context_precision"] = LLMContextPrecisionWithReference(
            llm=evaluator_llm
        )
    except Exception as exc:
        metric_errors["context_precision"] = str(exc)

    return metrics


# ---------------------------------------------------------
# Public entry point
# ---------------------------------------------------------

def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate response quality using RAGAS metrics.

    Args:
        question:  the user's question
        answer:    the RAG system's answer
        contexts:  the retrieved text chunks given to the LLM
        reference: optional ground-truth answer. BLEU, ROUGE and
                   context precision are only computed when it is given.

    Returns:
        A dict of metric name -> float score. If some metrics failed,
        a "metric_errors" dict explains why. If nothing could be scored,
        a dict with a single "error" key is returned.
    """

    if not RAGAS_AVAILABLE:
        return {"error": f"RAGAS not available: {RAGAS_IMPORT_ERROR}"}

    validation_error = _validate_inputs(question, answer, contexts, reference)

    if validation_error:
        return {"error": f"Invalid input: {validation_error}"}

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return {"error": "OPENAI_API_KEY is not set"}

    use_reference = reference is not None
    metric_errors: Dict[str, str] = {}

    try:
        metrics = _build_metrics(api_key, use_reference, metric_errors)
    except Exception as exc:
        return {"error": f"Could not set up evaluator: {exc}"}

    # Only real ground truth is ever used as the reference. Without one,
    # the reference fields stay empty and reference metrics are skipped.
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )

    async def run_evaluation() -> Dict[str, float]:
        results: Dict[str, float] = {}

        for name, metric in metrics.items():
            try:
                score = await metric.single_turn_ascore(sample)
                results[name] = float(score)
            except Exception as exc:
                metric_errors[name] = str(exc)

        return results

    try:
        scores: Dict[str, Any] = asyncio.run(run_evaluation())
    except Exception as exc:
        return {"error": f"Evaluation failed: {exc}"}

    if not scores:
        return {
            "error": "No metric could be computed: "
            + "; ".join(f"{k}: {v}" for k, v in metric_errors.items())
        }

    if metric_errors:
        scores["metric_errors"] = metric_errors

    return scores


if __name__ == "__main__":
    print(
        evaluate_response_quality(
            question="What went wrong on Apollo 13?",
            answer="An oxygen tank exploded in the service module.",
            contexts=[
                "On Apollo 13, oxygen tank number two exploded "
                "in the service module, crippling the spacecraft."
            ],
            reference=(
                "Oxygen tank No. 2 in the service module exploded, "
                "forcing the crew to abort the lunar landing."
            ),
        )
    )