import logging
import time
from typing import Any

from config import (
    QUESTION_FETCH_K,
    QUESTION_MAX_THRESHOLD,
    QUESTION_MIN_THRESHOLD,
    QUESTION_RESULT_K,
)
from data.collections import Collections
from service.retrieval.dto.retrieval_dto import Question, Retrieval

logger = logging.getLogger("rag_agent")


class RetrievalService:
    def __init__(self, collection: Collections):
        self.questions: Any = collection.questions

    @staticmethod
    def _query_with_retry(
        collection: Any, query: str, n_results: int
    ) -> dict[str, Any]:
        for attempt in range(3):
            try:
                # Result is sorted by ascending-ordered distance
                result: dict[str, Any] = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
                return result
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                logger.error(f"ChromaDB query failed: {e}")
                break

        # If all retries fail, return an empty result
        return {"ids": []}

    @staticmethod
    def _extract_question_stem(query: str) -> str:
        return query.split("\nA)")[0].strip()

    def search(self, query: str, extra: dict[str, Any] | None = None) -> Retrieval:
        question_stem = self._extract_question_stem(query)
        q_result = self._query_with_retry(
            self.questions, question_stem, QUESTION_FETCH_K
        )

        questions: list[Question] = []
        if q_result["ids"]:
            for doc, meta, dist in zip(
                q_result["documents"][0],
                q_result["metadatas"][0],
                q_result["distances"][0],
                strict=True,
            ):
                if QUESTION_MIN_THRESHOLD <= dist <= QUESTION_MAX_THRESHOLD:
                    questions.append(
                        Question(
                            document=doc,
                            A=meta["A"],
                            B=meta["B"],
                            C=meta["C"],
                            D=meta["D"],
                            answer=meta["answer"],
                            category=meta.get("category", "Unknown"),
                            distance=dist,
                        )
                    )
                    if len(questions) >= QUESTION_RESULT_K:
                        break

        q_dists = [f"{q.distance:.4f}" for q in questions]
        logger.debug(
            "search: %d questions (dist=%s)",
            len(questions),
            q_dists,
            extra=extra or {},  # pyright: ignore[reportCallIssue]
        )

        return Retrieval(questions=questions)


if __name__ == "__main__":
    from ecs_logging import StdlibFormatter

    from data.loader import setup_database

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
    service: RetrievalService = RetrievalService(collections)
    query_result: Retrieval = service.search(
        query="""국내 치안여건과 경찰의 역할에 관한 설명으로 옳지 않은 것은?
        A) 범죄의 양적ㆍ질적 심화로 인해 경찰은 역할한계에 직면하고 있다.
        B) 한국은 자치경찰제도를 운영하고 있지 않다.
        C) 경찰 1인당 담당하는 시민의 비율이 선진국에 비해 높은 편이다.
        D) 경찰은 민간경비와 마찬가지로 1차적으로 범죄예방에 초점을 두고 대응하고 있다.
        """
    )
    assert query_result is not None, "Retrieval result is None"
    logger.info(f"questions: {len(query_result.questions)}")
    for i, q in enumerate(query_result.questions):
        logger.info(f"Q[{i}] dist={q.distance:.4f}: {q.document[:80]}...")
