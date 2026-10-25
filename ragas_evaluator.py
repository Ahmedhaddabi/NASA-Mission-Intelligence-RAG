import os
import asyncio
from typing import Dict, List

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore,
    )
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings

    RAGAS_AVAILABLE = True
    RAGAS_IMPORT_ERROR = ""

except ImportError as exc:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(exc)


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, float]:
    """
    Evaluate response quality using RAGAS metrics.
    """

    if not RAGAS_AVAILABLE:
        return {"error": f"RAGAS not available: {RAGAS_IMPORT_ERROR}"}

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return {"error": "OPENAI_API_KEY is not set"}

    # ---------------------------------------------------------
    # Create evaluator LLM
    # ---------------------------------------------------------

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            base_url="https://openai.vocareum.com/v1",
            temperature=0,
        )
    )

    # ---------------------------------------------------------
    # Create evaluator embeddings
    # ---------------------------------------------------------

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
            base_url="https://openai.vocareum.com/v1",
        )
    )

    # ---------------------------------------------------------
    # Define RAGAS metrics
    # ---------------------------------------------------------

    response_relevancy = ResponseRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    faithfulness = Faithfulness(
        llm=evaluator_llm,
    )

    try:
        bleu_score = BleuScore()

        rouge_score = RougeScore()

        context_precision = (
            NonLLMContextPrecisionWithReference()
        )

    except ImportError as exc:
        return {
            "error": str(exc)
        }

    # ---------------------------------------------------------
    # Create evaluation sample
    # ---------------------------------------------------------

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=answer,
        reference_contexts=contexts,
    )

    # ---------------------------------------------------------
    # Evaluate metrics
    # ---------------------------------------------------------

    async def run_evaluation():
        results = {}

        results["response_relevancy"] = await (
            response_relevancy.single_turn_ascore(
                sample
            )
        )

        results["faithfulness"] = await (
            faithfulness.single_turn_ascore(
                sample
            )
        )

        results["bleu_score"] = await (
            bleu_score.single_turn_ascore(
                sample
            )
        )

        results["rouge_score"] = await (
            rouge_score.single_turn_ascore(
                sample
            )
        )

        results["context_precision"] = await (
            context_precision.single_turn_ascore(
                sample
            )
        )

        return results

    try:
        results = asyncio.run(
            run_evaluation()
        )

        return {
            key: float(value)
            for key, value in results.items()
        }

    except Exception as exc:
        return {
            "error": str(exc)
        }


if __name__ == "__main__":
    print(
        evaluate_response_quality(
            question="What went wrong on Apollo 13?",
            answer="An oxygen tank exploded in the service module.",
            contexts=[
                "On Apollo 13, oxygen tank number two exploded "
                "in the service module, crippling the spacecraft."
            ],
        )
    )