#!/usr/bin/env python3
"""
ChromaDB embedding pipeline for NASA space mission text files.

Supported data sources:
- Apollo 11
- Apollo 13
- Challenger
"""

import argparse
import hashlib
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from openai import OpenAI


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chroma_embedding_text_only.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """
    Pipeline for creating a persistent ChromaDB collection containing
    NASA mission text documents and OpenAI embeddings.
    """

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialise the embedding pipeline.

        Args:
            openai_api_key:
                OpenAI API key.

            chroma_persist_directory:
                Directory in which ChromaDB data will be persisted.

            collection_name:
                Name of the ChromaDB collection.

            embedding_model:
                OpenAI embedding model.

            chunk_size:
                Maximum chunk size in characters.

            chunk_overlap:
                Character overlap between consecutive chunks.
        """

        if not openai_api_key:
            raise ValueError("An OpenAI API key is required")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.openai_client = OpenAI(
            base_url="https://openai.vocareum.com/v1",
            api_key=openai_api_key,
        )

        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            ),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "NASA space mission text documents",
                "embedding_model": self.embedding_model,
            },
        )

        logger.info(
            "Initialised collection '%s' with %s existing documents",
            self.collection_name,
            self.collection.count(),
        )

    # -----------------------------------------------------------------------
    # Text handling
    # -----------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """
        Clean document text before chunking.
        """

        if not text:
            return ""

        text = text.replace("\x00", " ")

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def chunk_text(
        self,
        text: str,
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        """

        text = self.clean_text(text)

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(
                start + self.chunk_size,
                text_length,
            )

            if end < text_length:
                search_start = max(
                    start,
                    end - 200,
                )

                break_positions = []

                for separator in [
                    "\n\n",
                    "\n",
                    ". ",
                    "? ",
                    "! ",
                    " ",
                ]:
                    position = text.rfind(
                        separator,
                        search_start,
                        end,
                    )

                    if position != -1:
                        position += len(separator)
                        break_positions.append(position)

                if break_positions:
                    candidate_end = max(break_positions)

                    if candidate_end > start:
                        end = candidate_end

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    # -----------------------------------------------------------------------
    # File loading
    # -----------------------------------------------------------------------

    def read_text_file(
        self,
        file_path: Path,
    ) -> str:
        """
        Read a text file while handling common encoding problems.
        """

        encodings = [
            "utf-8",
            "utf-8-sig",
            "latin-1",
        ]

        for encoding in encodings:
            try:
                return file_path.read_text(
                    encoding=encoding
                )

            except UnicodeDecodeError:
                continue

            except Exception as exc:
                logger.error(
                    "Unable to read %s: %s",
                    file_path,
                    exc,
                )
                return ""

        logger.error(
            "Unable to decode file %s",
            file_path,
        )

        return ""

    def determine_source(
        self,
        file_path: Path,
        root_path: Path,
    ) -> str:
        """
        Determine mission/source name from a file's relative path.
        """

        try:
            relative = file_path.relative_to(root_path)

            if len(relative.parts) > 1:
                return relative.parts[0]

        except ValueError:
            pass

        return file_path.parent.name

    def create_metadata(
        self,
        file_path: Path,
        root_path: Path,
        chunk_index: int,
        total_chunks: int,
    ) -> Dict[str, Any]:
        """
        Create ChromaDB-safe metadata for a document chunk.
        """

        source = self.determine_source(
            file_path,
            root_path,
        )

        try:
            relative_path = str(
                file_path.relative_to(root_path)
            )
        except ValueError:
            relative_path = str(file_path)

        metadata: Dict[str, Any] = {
            "source": source,
            "mission": source,
            "file_name": file_path.name,
            "file_path": relative_path,
            "chunk_index": int(chunk_index),
            "total_chunks": int(total_chunks),
            "file_extension": file_path.suffix.lower(),
        }

        try:
            metadata["file_size"] = int(
                file_path.stat().st_size
            )

            metadata["modified_time"] = datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).isoformat()

        except OSError:
            metadata["file_size"] = 0
            metadata["modified_time"] = ""

        return metadata

    def process_text_file(
        self,
        file_path: Path,
        root_path: Path,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read, clean and chunk a text document.
        """

        text = self.read_text_file(file_path)

        if not text.strip():
            logger.warning(
                "Skipping empty file: %s",
                file_path,
            )
            return []

        chunks = self.chunk_text(text)

        documents: List[
            Tuple[str, Dict[str, Any]]
        ] = []

        total_chunks = len(chunks)

        for index, chunk in enumerate(chunks):
            metadata = self.create_metadata(
                file_path=file_path,
                root_path=root_path,
                chunk_index=index,
                total_chunks=total_chunks,
            )

            documents.append(
                (
                    chunk,
                    metadata,
                )
            )

        return documents

    # -----------------------------------------------------------------------
    # IDs
    # -----------------------------------------------------------------------

    def generate_document_id(
        self,
        file_path: Path,
        metadata: Dict[str, Any],
    ) -> str:
        """
        Generate a stable ID for a file chunk.
        """

        relative_path = metadata.get(
            "file_path",
            str(file_path),
        )

        chunk_index = metadata.get(
            "chunk_index",
            0,
        )

        raw_id = (
            f"{relative_path}::{chunk_index}"
        )

        digest = hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

        return digest

    # -----------------------------------------------------------------------
    # Embeddings
    # -----------------------------------------------------------------------

    def create_embedding(
        self,
        text: str,
        max_retries: int = 3,
    ) -> List[float]:
        """
        Generate an OpenAI embedding for text.
        """

        if not text.strip():
            raise ValueError(
                "Cannot create embedding for empty text"
            )

        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = (
                    self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=text,
                    )
                )

                return response.data[0].embedding

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Embedding request failed "
                    "(attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )

                if attempt < max_retries - 1:
                    time.sleep(
                        2 ** attempt
                    )

        raise RuntimeError(
            "Failed to create embedding "
            f"after {max_retries} attempts: "
            f"{last_error}"
        )

    def create_embeddings(
        self,
        texts: List[str],
        max_retries: int = 3,
    ) -> List[List[float]]:
        """
        Create embeddings for a batch of texts.
        """

        if not texts:
            return []

        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = (
                    self.openai_client.embeddings.create(
                        model=self.embedding_model,
                        input=texts,
                    )
                )

                ordered_data = sorted(
                    response.data,
                    key=lambda item: item.index,
                )

                return [
                    item.embedding
                    for item in ordered_data
                ]

            except Exception as exc:
                last_error = exc

                logger.warning(
                    "Batch embedding request failed "
                    "(attempt %s/%s): %s",
                    attempt + 1,
                    max_retries,
                    exc,
                )

                if attempt < max_retries - 1:
                    time.sleep(
                        2 ** attempt
                    )

        raise RuntimeError(
            "Failed to create embeddings "
            f"after {max_retries} attempts: "
            f"{last_error}"
        )

    # -----------------------------------------------------------------------
    # Collection helpers
    # -----------------------------------------------------------------------

    def check_document_exists(
        self,
        document_id: str,
    ) -> bool:
        """
        Check whether a chunk already exists.
        """

        try:
            result = self.collection.get(
                ids=[document_id]
            )

            return bool(
                result.get("ids")
            )

        except Exception:
            return False

    def update_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> bool:
        """
        Update an existing document in ChromaDB.
        """

        try:
            embedding = self.create_embedding(text)

            self.collection.update(
                ids=[document_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding],
            )

            return True

        except Exception as exc:
            logger.error(
                "Failed to update document %s: %s",
                document_id,
                exc,
            )

            return False

    def get_file_documents(
        self,
        relative_path: str,
    ) -> List[str]:
        """
        Find existing chunk IDs for a particular file, matched on the
        relative path stored in each chunk's metadata.
        """

        try:
            result = self.collection.get(
                where={
                    "file_path": relative_path
                },
                include=[],
            )

        except Exception as exc:
            logger.error(
                "Unable to inspect collection: %s",
                exc,
            )
            return []

        return list(
            result.get("ids") or []
        )

    # -----------------------------------------------------------------------
    # Document insertion
    # -----------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[
            Tuple[str, Dict[str, Any]]
        ],
        file_path: Path,
        batch_size: int = 20,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        """
        Add document chunks to ChromaDB.

        update_mode:
            skip:
                Leave chunks that already exist.

            update:
                Update chunks that already exist with new text,
                metadata and embeddings.

            replace:
                Delete existing chunks for the file,
                then add the newly generated chunks.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        if update_mode not in {
            "skip",
            "update",
            "replace",
        }:
            raise ValueError(
                "update_mode must be skip, update or replace"
            )

        stats = {
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "failed": 0,
        }

        if not documents:
            return stats

        if update_mode == "replace":
            relative_path = documents[0][1].get(
                "file_path",
                str(file_path),
            )

            old_document_ids = (
                self.get_file_documents(
                    relative_path
                )
            )

            if old_document_ids:
                self.collection.delete(
                    ids=old_document_ids
                )

                logger.info(
                    "Deleted %s existing chunks for %s",
                    len(old_document_ids),
                    file_path.name,
                )

        for batch_start in range(
            0,
            len(documents),
            batch_size,
        ):
            batch = documents[
                batch_start:
                batch_start + batch_size
            ]

            ids_to_add: List[str] = []
            texts_to_add: List[str] = []
            metadatas_to_add: List[
                Dict[str, Any]
            ] = []

            for text, metadata in batch:
                document_id = (
                    self.generate_document_id(
                        file_path,
                        metadata,
                    )
                )

                exists = (
                    self.check_document_exists(
                        document_id
                    )
                )

                if (
                    exists
                    and update_mode == "skip"
                ):
                    stats["skipped"] += 1
                    continue

                if (
                    exists
                    and update_mode == "update"
                ):
                    successful = (
                        self.update_document(
                            document_id,
                            text,
                            metadata,
                        )
                    )

                    if successful:
                        stats["updated"] += 1
                    else:
                        stats["failed"] += 1

                    continue

                ids_to_add.append(
                    document_id
                )
                texts_to_add.append(
                    text
                )
                metadatas_to_add.append(
                    metadata
                )

            if not texts_to_add:
                continue

            try:
                embeddings_to_add = (
                    self.create_embeddings(
                        texts_to_add
                    )
                )

                self.collection.add(
                    ids=ids_to_add,
                    documents=texts_to_add,
                    metadatas=metadatas_to_add,
                    embeddings=embeddings_to_add,
                )

                stats["added"] += len(
                    ids_to_add
                )

            except Exception as exc:
                logger.error(
                    "Failed to add batch for %s: %s",
                    file_path,
                    exc,
                )

                stats["failed"] += len(
                    ids_to_add
                )

        return stats

    # -----------------------------------------------------------------------
    # Pipeline execution
    # -----------------------------------------------------------------------

    def find_text_files(
        self,
        data_path: Path,
    ) -> List[Path]:
        """
        Recursively discover text files.
        """

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data path does not exist: {data_path}"
            )

        if data_path.is_file():
            if (
                data_path.suffix.lower()
                == ".txt"
            ):
                return [data_path]

            return []

        files = sorted(
            path
            for path in data_path.rglob("*.txt")
            if path.is_file()
        )

        return files

    def process_directory(
        self,
        data_path: str,
        batch_size: int = 20,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        """
        Process all text files under a directory.
        """

        root_path = Path(
            data_path
        ).expanduser().resolve()

        text_files = self.find_text_files(
            root_path
        )

        # When a single file is passed, use its folder as the root so the
        # relative path (and therefore the chunk IDs) stays unique per file.
        if root_path.is_file():
            root_path = root_path.parent

        logger.info(
            "Found %s text files in %s",
            len(text_files),
            root_path,
        )

        totals = {
            "files": 0,
            "chunks": 0,
            "added": 0,
            "updated": 0,
            "skipped": 0,
            "failed": 0,
        }

        for file_path in text_files:
            logger.info(
                "Processing %s",
                file_path,
            )

            try:
                documents = (
                    self.process_text_file(
                        file_path=file_path,
                        root_path=root_path,
                    )
                )

                totals["files"] += 1
                totals["chunks"] += len(
                    documents
                )

                stats = self.add_documents(
                    documents=documents,
                    file_path=file_path,
                    batch_size=batch_size,
                    update_mode=update_mode,
                )

                for key in [
                    "added",
                    "updated",
                    "skipped",
                    "failed",
                ]:
                    totals[key] += stats[key]

                logger.info(
                    "%s -> chunks=%s "
                    "added=%s updated=%s "
                    "skipped=%s failed=%s",
                    file_path.name,
                    len(documents),
                    stats["added"],
                    stats["updated"],
                    stats["skipped"],
                    stats["failed"],
                )

            except Exception as exc:
                totals["failed"] += 1

                logger.exception(
                    "Failed processing %s: %s",
                    file_path,
                    exc,
                )

        logger.info(
            "Processing complete: %s",
            totals,
        )

        return totals

    # -----------------------------------------------------------------------
    # Querying
    # -----------------------------------------------------------------------

    def query_collection(
        self,
        query: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform semantic search against the collection.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        if n_results <= 0:
            raise ValueError(
                "n_results must be greater than zero"
            )

        count = self.collection.count()

        if count == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        query_embedding = (
            self.create_embedding(query)
        )

        result_count = min(
            n_results,
            count,
        )

        return self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=result_count,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    def print_query_results(
        self,
        result: Dict[str, Any],
    ) -> None:
        """
        Print semantic search results.
        """

        documents = (
            result.get("documents")
            or [[]]
        )[0]

        metadatas = (
            result.get("metadatas")
            or [[]]
        )[0]

        distances = (
            result.get("distances")
            or [[]]
        )[0]

        if not documents:
            print(
                "No matching documents found."
            )
            return

        for index, document in enumerate(
            documents,
            start=1,
        ):
            metadata = (
                metadatas[index - 1]
                if index - 1
                < len(metadatas)
                else {}
            )

            distance = (
                distances[index - 1]
                if index - 1
                < len(distances)
                else None
            )

            print(
                f"\n--- Result {index} ---"
            )

            print(
                "Source:",
                metadata.get(
                    "source",
                    "unknown",
                ),
            )

            print(
                "File:",
                metadata.get(
                    "file_name",
                    "unknown",
                ),
            )

            print(
                "Chunk:",
                metadata.get(
                    "chunk_index",
                    "unknown",
                ),
            )

            if distance is not None:
                print(
                    "Distance:",
                    distance,
                )

            print(
                "\n",
                document,
            )

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_collection_stats(
        self,
    ) -> Dict[str, Any]:
        """
        Return basic collection statistics.
        """

        total_documents = (
            self.collection.count()
        )

        stats: Dict[str, Any] = {
            "collection_name":
                self.collection_name,
            "embedding_model":
                self.embedding_model,
            "total_documents":
                total_documents,
            "sources": {},
        }

        if total_documents == 0:
            return stats

        try:
            result = self.collection.get(
                include=["metadatas"]
            )

            metadatas = (
                result.get("metadatas")
                or []
            )

            source_counts: Dict[
                str,
                int
            ] = {}

            for metadata in metadatas:
                if not metadata:
                    continue

                source = metadata.get(
                    "source",
                    "unknown",
                )

                source_counts[source] = (
                    source_counts.get(
                        source,
                        0,
                    )
                    + 1
                )

            stats["sources"] = (
                source_counts
            )

        except Exception as exc:
            logger.error(
                "Unable to calculate "
                "collection statistics: %s",
                exc,
            )

        return stats

    def delete_source(
        self,
        source: str,
    ) -> int:
        """
        Delete all chunks associated with a source.
        """

        result = self.collection.get(
            where={
                "source": source
            }
        )

        document_ids = (
            result.get("ids")
            or []
        )

        if not document_ids:
            logger.info(
                "No documents found for source '%s'",
                source,
            )
            return 0

        self.collection.delete(
            ids=document_ids
        )

        logger.info(
            "Deleted %s documents for source '%s'",
            len(document_ids),
            source,
        )

        return len(
            document_ids
        )


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create and query a ChromaDB "
            "collection for NASA mission "
            "text documents."
        )
    )

    parser.add_argument(
        "--data-path",
        default="./data_text",
        help=(
            "Directory containing NASA "
            "mission text files"
        ),
    )

    parser.add_argument(
        "--openai-key",
        default=os.getenv(
            "OPENAI_API_KEY"
        ),
        help=(
            "OpenAI API key. "
            "Defaults to OPENAI_API_KEY."
        ),
    )

    parser.add_argument(
        "--chroma-dir",
        default="./chroma_db",
        help=(
            "Directory used to persist "
            "ChromaDB"
        ),
    )

    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="ChromaDB collection name",
    )

    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help=(
            "Number of overlapping "
            "characters between chunks"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help=(
            "Number of chunks sent per "
            "embedding batch"
        ),
    )

    parser.add_argument(
        "--update-mode",
        choices=[
            "skip",
            "update",
            "replace",
        ],
        default="skip",
        help=(
            "Behaviour when chunks already "
            "exist"
        ),
    )

    parser.add_argument(
        "--test-query",
        help=(
            "Run a semantic search after "
            "processing"
        ),
    )

    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help=(
            "Number of semantic-search "
            "results"
        ),
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help=(
            "Print collection statistics "
            "without processing documents"
        ),
    )

    parser.add_argument(
        "--delete-source",
        help=(
            "Delete all chunks belonging "
            "to the supplied source"
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    if not args.openai_key:
        parser.error(
            "An OpenAI API key is required. "
            "Set OPENAI_API_KEY or provide "
            "--openai-key."
        )

    pipeline = (
        ChromaEmbeddingPipelineTextOnly(
            openai_api_key=args.openai_key,
            chroma_persist_directory=(
                args.chroma_dir
            ),
            collection_name=(
                args.collection_name
            ),
            embedding_model=(
                args.embedding_model
            ),
            chunk_size=args.chunk_size,
            chunk_overlap=(
                args.chunk_overlap
            ),
        )
    )

    if args.delete_source:
        deleted = pipeline.delete_source(
            args.delete_source
        )

        print(
            f"Deleted {deleted} chunks "
            f"for source "
            f"'{args.delete_source}'."
        )

        return

    if args.stats_only:
        stats = (
            pipeline.get_collection_stats()
        )

        print("\nCollection statistics")

        for key, value in stats.items():
            print(
                f"{key}: {value}"
            )

        return

    totals = pipeline.process_directory(
        data_path=args.data_path,
        batch_size=args.batch_size,
        update_mode=args.update_mode,
    )

    print("\nEmbedding pipeline complete")

    for key, value in totals.items():
        print(
            f"{key}: {value}"
        )

    stats = (
        pipeline.get_collection_stats()
    )

    print("\nCollection statistics")

    for key, value in stats.items():
        print(
            f"{key}: {value}"
        )

    if args.test_query:
        print(
            f"\nQuery: {args.test_query}"
        )

        result = pipeline.query_collection(
            query=args.test_query,
            n_results=args.n_results,
        )

        pipeline.print_query_results(
            result
        )


if __name__ == "__main__":
    main()