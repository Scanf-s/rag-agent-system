import csv
import logging
import os
import time
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from ecs_logging import StdlibFormatter

from config import EMBEDDING_MODEL, LOADER_BATCH_SIZE, LOADER_MAX_RETRIES
from data.collections import Collections

ANSWER_MAP: dict[str, str] = {"1": "A", "2": "B", "3": "C", "4": "D"}

# Load environment variables
load_dotenv()
ENV: str = os.environ.get("ENV", "INFO")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
logger = logging.getLogger("rag_agent")


def _add_with_retry(
    collection: Any,
    documents: list[str],
    metadatas: list[dict[str, str]],
    ids: list[str],
) -> None:
    """
    Perform a batch add operation with retry logic. (exponential backoff)
    :param collection: the collection to add to
    :param documents: the documents to add
    :param metadatas: the metadata to add
    :param ids: the ids to add
    :return: None
    """
    for attempt in range(1, LOADER_MAX_RETRIES + 1):
        try:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            return
        except Exception as e:
            if "429" in str(e) and attempt < LOADER_MAX_RETRIES:
                wait = 2**attempt
                logger.warning(f"rate limit exceeded. Wait for {wait}s and try again.")
                time.sleep(wait)
            else:
                logger.error(f"Failed to add documents: {e}")
                exit(1)


def load_train_questions(collection: Any) -> bool:
    """
    This function loads the training questions dataset and creates a vector database.
    :param collection: The chroma collection to add the data to.
    :return: True if the data was loaded successfully, False otherwise.
    """
    logger.info("loading train questions")
    try:
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []
        ids: list[str] = []

        with open("data/train.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                doc = f"{row['question']}"
                documents.append(doc)
                metadatas.append(
                    {
                        "A": row["A"],
                        "B": row["B"],
                        "C": row["C"],
                        "D": row["D"],
                        "answer": ANSWER_MAP[row["answer"]],
                        "category": row["Category"],
                    }
                )
                ids.append(f"train_{i}")

                if len(documents) == LOADER_BATCH_SIZE:
                    _add_with_retry(
                        collection=collection,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                    )
                    logger.info(f"#{i} train_questions loaded into chroma database")
                    documents, metadatas, ids = [], [], []

        if documents:
            _add_with_retry(
                collection=collection, documents=documents, metadatas=metadatas, ids=ids
            )

        logger.info(f"loaded {collection.count()} train_questions successfully")
        return True
    except Exception as e:
        logger.error(f"failed to load train_questions: {e}")
        return False


def setup_database() -> Collections:
    chroma_client = chromadb.PersistentClient(path="./data/chroma_db")

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY, model_name=EMBEDDING_MODEL
    )

    question_collection: Any = chroma_client.get_or_create_collection(
        name="questions",
        embedding_function=openai_ef,  # pyright: ignore[reportArgumentType]
    )

    logger.info("database initialize completed")
    return Collections(questions=question_collection)


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        StdlibFormatter(
            exclude_fields=["ecs.version", "process", "log.origin", "original"],
            ensure_ascii=False,
        )
    )
    logger.addHandler(handler)
    logger.info("data loading started.")
    collections: Collections = setup_database()

    result = load_train_questions(collections.questions)
    if not result:
        logger.error("train data loading failed")
        os.remove("./data/chroma_db")
        exit(1)
    logger.info(f"train data loading task completed: {result}")
    logger.info("all data loading tasks were completed")
