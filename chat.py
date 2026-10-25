#!/usr/bin/env python3
"""
NASA RAG Chat with RAGAS Evaluation Integration

Streamlit application for chatting with NASA mission documents using
RAG retrieval, LLM generation, and RAGAS evaluation.
"""

import os
from typing import Dict, List, Optional

import streamlit as st

import ragas_evaluator
import rag_client
import llm_client


# ---------------------------------------------------------------------------
# RAGAS availability
# ---------------------------------------------------------------------------

try:
    from ragas import SingleTurnSample
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NASA RAG Chat with Evaluation",
    page_icon="🚀",
    layout="wide"
)


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def discover_chroma_backends() -> Dict[str, Dict]:
    """Discover available ChromaDB backends."""

    return rag_client.discover_chroma_backends()


@st.cache_resource
def initialize_rag_system(
    chroma_dir: str,
    collection_name: str
):
    """Initialise the selected ChromaDB collection."""

    return rag_client.initialize_rag_system(
        chroma_dir,
        collection_name
    )


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """Retrieve relevant NASA documents."""

    try:
        return rag_client.retrieve_documents(
            collection,
            query,
            n_results,
            mission_filter
        )

    except Exception as error:
        st.error(
            f"Error retrieving documents: {error}"
        )
        return None


def format_context(
    documents: List[str],
    metadatas: List[Dict]
) -> str:
    """Format retrieved documents into LLM context."""

    return rag_client.format_context(
        documents,
        metadatas
    )


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str
) -> str:
    """Generate a grounded response using the LLM."""

    try:
        return llm_client.generate_response(
            openai_key,
            user_message,
            context,
            conversation_history,
            model
        )

    except Exception as error:
        return (
            "Error generating response: "
            f"{error}"
        )


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str]
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS."""

    try:
        return (
            ragas_evaluator.evaluate_response_quality(
                question,
                answer,
                contexts
            )
        )

    except Exception as error:
        return {
            "error":
                f"Evaluation failed: {error}"
        }


def display_evaluation_metrics(
    scores: Dict[str, float]
):
    """Display evaluation metrics."""

    if "error" in scores:
        st.sidebar.error(
            f"Evaluation Error: {scores['error']}"
        )
        return

    st.sidebar.subheader(
        "📊 Response Quality"
    )

    for metric_name, score in scores.items():

        if not isinstance(
            score,
            (int, float)
        ):
            continue

        st.sidebar.metric(
            label=metric_name.replace(
                "_",
                " "
            ).title(),
            value=f"{score:.3f}"
        )

        # Streamlit progress values must
        # remain between 0 and 1.
        safe_score = max(
            0.0,
            min(1.0, float(score))
        )

        st.sidebar.progress(
            safe_score
        )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def main():

    st.title(
        "🚀 NASA Space Mission Chat with Evaluation"
    )

    st.markdown(
        "Chat with AI about NASA space missions "
        "using retrieval-augmented generation."
    )

    # ---------------------------------------------------------
    # Session state
    # ---------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "current_backend" not in st.session_state:
        st.session_state.current_backend = None

    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None

    if "last_contexts" not in st.session_state:
        st.session_state.last_contexts = []

    # ---------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------

    with st.sidebar:

        st.header(
            "🔧 Configuration"
        )

        # -----------------------------------------------------
        # ChromaDB
        # -----------------------------------------------------

        with st.spinner(
            "Discovering ChromaDB backends..."
        ):
            available_backends = (
                discover_chroma_backends()
            )

        if not available_backends:

            st.error(
                "No ChromaDB backends found."
            )

            st.info(
                "Run embedding_pipeline.py "
                "to build the NASA collection."
            )

            st.stop()

        st.subheader(
            "📊 ChromaDB Backend"
        )

        backend_options = {
            key: value["display_name"]
            for key, value
            in available_backends.items()
            if value.get("collection")
        }

        if not backend_options:
            st.error(
                "No usable ChromaDB collections found."
            )
            st.stop()

        selected_backend_key = st.selectbox(
            "Select Document Collection",
            options=list(
                backend_options.keys()
            ),
            format_func=lambda value:
                backend_options[value],
            help=(
                "Choose which NASA document "
                "collection to search."
            )
        )

        selected_backend = (
            available_backends[
                selected_backend_key
            ]
        )

        # -----------------------------------------------------
        # API key
        # -----------------------------------------------------

        st.subheader(
            "🔑 OpenAI Settings"
        )

        openai_key = st.text_input(
            "Vocareum OpenAI API Key",
            type="password",
            value=os.getenv(
                "OPENAI_API_KEY",
                ""
            ),
            help=(
                "Uses the OPENAI_API_KEY "
                "environment variable by default."
            )
        )

        if not openai_key:

            st.warning(
                "Please provide your "
                "Vocareum OpenAI API key."
            )

            st.stop()

        # Make the key available to rag_client
        # and ragas_evaluator.
        os.environ[
            "OPENAI_API_KEY"
        ] = openai_key

        # Ensure the correct Vocareum endpoint.
        os.environ[
            "OPENAI_BASE_URL"
        ] = (
            "https://openai.vocareum.com/v1"
        )

        # -----------------------------------------------------
        # Model
        # -----------------------------------------------------

        model_choice = st.selectbox(
            "LLM Model",
            options=[
                "gpt-4o-mini"
            ],
            help=(
                "Select the model used to "
                "generate responses."
            )
        )

        # -----------------------------------------------------
        # Retrieval
        # -----------------------------------------------------

        st.subheader(
            "🔍 Retrieval Settings"
        )

        n_docs = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=10,
            value=3
        )

        mission_filter = st.selectbox(
            "Mission",
            options=[
                "All Missions",
                "Apollo 11",
                "Apollo 13",
                "Challenger"
            ]
        )

        # -----------------------------------------------------
        # Evaluation
        # -----------------------------------------------------

        st.subheader(
            "📊 Evaluation Settings"
        )

        enable_evaluation = st.checkbox(
            "Enable RAGAS Evaluation",
            value=RAGAS_AVAILABLE
        )

        if not RAGAS_AVAILABLE:
            st.warning(
                "RAGAS is not available."
            )

        # -----------------------------------------------------
        # Backend state
        # -----------------------------------------------------

        if (
            st.session_state.current_backend
            != selected_backend_key
        ):
            st.session_state.current_backend = (
                selected_backend_key
            )

            initialize_rag_system.clear()

    # ---------------------------------------------------------
    # Initialise ChromaDB
    # ---------------------------------------------------------

    try:

        with st.spinner(
            "Initializing RAG system..."
        ):

            collection = (
                initialize_rag_system(
                    selected_backend["path"],
                    selected_backend[
                        "collection"
                    ]
                )
            )

    except Exception as error:

        st.error(
            "Failed to initialize RAG system: "
            f"{error}"
        )

        st.stop()

    # ---------------------------------------------------------
    # Collection information
    # ---------------------------------------------------------

    st.sidebar.success(
        f"Collection loaded: "
        f"{collection.count()} documents"
    )

    # ---------------------------------------------------------
    # Previous evaluation
    # ---------------------------------------------------------

    if (
        st.session_state.last_evaluation
        and enable_evaluation
    ):
        display_evaluation_metrics(
            st.session_state.last_evaluation
        )

    # ---------------------------------------------------------
    # Existing conversation
    # ---------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    # ---------------------------------------------------------
    # User question
    # ---------------------------------------------------------

    prompt = st.chat_input(
        "Ask about NASA space missions..."
    )

    if not prompt:
        return

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------------------------------------------------------
    # Retrieval + generation
    # ---------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching NASA documents..."
        ):

            docs_result = (
                retrieve_documents(
                    collection=collection,
                    query=prompt,
                    n_results=n_docs,
                    mission_filter=mission_filter
                )
            )

        context = ""
        contexts_list = []

        if (
            docs_result
            and docs_result.get(
                "documents"
            )
            and docs_result["documents"][0]
        ):

            documents = (
                docs_result[
                    "documents"
                ][0]
            )

            metadatas = (
                docs_result.get(
                    "metadatas"
                )
                or [[]]
            )[0]

            contexts_list = documents

            context = format_context(
                documents,
                metadatas
            )

            st.session_state.last_contexts = (
                contexts_list
            )

        else:

            st.warning(
                "No relevant NASA documents "
                "were retrieved."
            )

        # -----------------------------------------------------
        # LLM response
        # -----------------------------------------------------

        with st.spinner(
            "Generating response..."
        ):

            response = generate_response(
                openai_key=openai_key,
                user_message=prompt,
                context=context,
                conversation_history=(
                    st.session_state.messages[
                        :-1
                    ]
                ),
                model=model_choice
            )

        st.markdown(
            response
        )

        # -----------------------------------------------------
        # RAGAS
        # -----------------------------------------------------

        if (
            enable_evaluation
            and RAGAS_AVAILABLE
            and contexts_list
            and not response.startswith(
                "Error generating response:"
            )
        ):

            with st.spinner(
                "Evaluating response quality..."
            ):

                evaluation_scores = (
                    evaluate_response_quality(
                        prompt,
                        response,
                        contexts_list
                    )
                )

                st.session_state.last_evaluation = (
                    evaluation_scores
                )

    # ---------------------------------------------------------
    # Save assistant response
    # ---------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    st.rerun()


if __name__ == "__main__":
    main()