import os
import re

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import Dict, List, Optional
from pathlib import Path


# Must match the model used by embedding_pipeline.py
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """Create the OpenAI client once."""

    global _openai_client

    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv(
                "OPENAI_BASE_URL",
                "https://openai.vocareum.com/v1"
            )
        )

    return _openai_client


def _normalise_mission(name: str) -> str:
    """
    Turn values such as:
    'Apollo 11', 'apollo_11', or 'APOLLO11'
    into 'apollo11'.
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower()
    )


def discover_chroma_backends() -> Dict[str, Dict]:
    """Discover available ChromaDB backends."""

    backends = {}
    current_dir = Path(".")

    chroma_dirs = [
        directory
        for directory in current_dir.iterdir()
        if directory.is_dir()
        and "chroma" in directory.name.lower()
    ]

    for directory in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(directory),
                settings=Settings(
                    anonymized_telemetry=False
                )
            )

            collections = client.list_collections()

            for collection_item in collections:

                if isinstance(
                    collection_item,
                    str
                ):
                    collection_name = (
                        collection_item
                    )

                    collection = (
                        client.get_collection(
                            name=collection_name
                        )
                    )

                else:
                    collection_name = (
                        collection_item.name
                    )

                    collection = (
                        collection_item
                    )

                backend_key = (
                    f"{directory.name}:"
                    f"{collection_name}"
                )

                try:
                    document_count = (
                        collection.count()
                    )
                except Exception:
                    document_count = 0

                backends[backend_key] = {
                    "path": str(directory),
                    "collection":
                        collection_name,
                    "display_name": (
                        f"{directory.name} / "
                        f"{collection_name} "
                        f"({document_count} documents)"
                    ),
                    "document_count":
                        document_count
                }

        except Exception as error:

            error_message = str(error)

            if len(error_message) > 80:
                error_message = (
                    error_message[:77]
                    + "..."
                )

            backend_key = (
                f"{directory.name}:error"
            )

            backends[backend_key] = {
                "path": str(directory),
                "collection": "",
                "display_name": (
                    f"{directory.name} "
                    f"(Unable to access: "
                    f"{error_message})"
                ),
                "document_count": 0
            }

    return backends


def initialize_rag_system(
    chroma_dir: str,
    collection_name: str
):
    """Initialise the selected ChromaDB backend."""

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    return client.get_collection(
        name=collection_name
    )


def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None
) -> Optional[Dict]:
    """
    Retrieve relevant documents using semantic search.

    Results are returned in ChromaDB similarity order.
    Duplicate chunks are removed while preserving
    that ranking.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    if n_results <= 0:
        raise ValueError(
            "n_results must be greater than zero."
        )

    where_filter = None

    if (
        mission_filter
        and mission_filter.strip().lower()
        not in {"all", "all missions"}
    ):
        mission_value = _normalise_mission(
            mission_filter
        )

        # Existing collections use "source" for the
        # mission identifier. This remains compatible
        # with those collections.
        where_filter = {
            "source": mission_value
        }

    count = collection.count()

    if count == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    # Use the same embedding model that was used
    # when the collection was created.
    embedding_model = (
        (collection.metadata or {}).get(
            "embedding_model"
        )
        or DEFAULT_EMBEDDING_MODEL
    )

    query_embedding = (
        _get_openai_client()
        .embeddings.create(
            model=embedding_model,
            input=query
        )
        .data[0]
        .embedding
    )

    query_parameters = {
        "query_embeddings": [
            query_embedding
        ],
        "n_results": min(
            n_results,
            count
        ),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if where_filter is not None:
        query_parameters[
            "where"
        ] = where_filter

    results = collection.query(
        **query_parameters
    )

    # --------------------------------------------------------
    # Deduplicate results
    # --------------------------------------------------------
    #
    # ChromaDB already returns results according to
    # similarity. We preserve that order while removing
    # duplicate text chunks.
    # --------------------------------------------------------

    documents = (
        results.get("documents")
        or [[]]
    )[0]

    metadatas = (
        results.get("metadatas")
        or [[]]
    )[0]

    distances = (
        results.get("distances")
        or [[]]
    )[0]

    ids = (
        results.get("ids")
        or [[]]
    )[0]

    seen_documents = set()

    unique_documents: List[str] = []
    unique_metadatas: List[Dict] = []
    unique_distances = []
    unique_ids: List[str] = []

    for index, document in enumerate(
        documents
    ):
        if not document:
            continue

        # Normalise whitespace before comparing
        # chunks so formatting differences do not
        # cause duplicates to pass through.
        document_key = " ".join(
            document.split()
        )

        if document_key in seen_documents:
            continue

        seen_documents.add(
            document_key
        )

        unique_documents.append(
            document
        )

        if index < len(metadatas):
            unique_metadatas.append(
                metadatas[index] or {}
            )
        else:
            unique_metadatas.append(
                {}
            )

        if index < len(distances):
            unique_distances.append(
                distances[index]
            )
        else:
            unique_distances.append(
                None
            )

        if index < len(ids):
            unique_ids.append(
                ids[index]
            )
        else:
            unique_ids.append(
                ""
            )

    return {
        "ids": [
            unique_ids
        ],
        "documents": [
            unique_documents
        ],
        "metadatas": [
            unique_metadatas
        ],
        "distances": [
            unique_distances
        ],
    }


def format_context(
    documents: List[str],
    metadatas: List[Dict]
) -> str:
    """
    Format retrieved NASA chunks into a single
    context string with source attribution.
    """

    if not documents:
        return ""

    context_parts = [
        "Relevant information retrieved "
        "from NASA mission documents:"
    ]

    for index, (
        document,
        metadata
    ) in enumerate(
        zip(
            documents,
            metadatas
        ),
        start=1
    ):

        metadata = metadata or {}

        # New collections can contain the explicit
        # "mission" field. Existing collections
        # fall back to "source".
        mission = (
            metadata.get("mission")
            or metadata.get("source")
            or "Unknown mission"
        )

        mission = str(
            mission
        ).replace(
            "_",
            " "
        ).title()

        source = (
            metadata.get("file_name")
            or metadata.get("source")
            or "Unknown source"
        )

        category = metadata.get(
            "category",
            "General"
        )

        category = str(
            category
        ).replace(
            "_",
            " "
        ).title()

        source_header = (
            f"\nSource {index} | "
            f"Mission: {mission} | "
            f"Category: {category} | "
            f"Source: {source}"
        )

        context_parts.append(
            source_header
        )

        maximum_document_length = 4000

        if (
            len(document)
            > maximum_document_length
        ):
            document_content = (
                document[
                    :maximum_document_length
                ]
                + "\n"
                "[Document chunk truncated]"
            )
        else:
            document_content = document

        context_parts.append(
            document_content
        )

    return "\n".join(
        context_parts
    )