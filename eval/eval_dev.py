import asyncio
import csv
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from ecs_logging import StdlibFormatter
from openai import AsyncOpenAI

from data.loader import ANSWER_MAP, setup_database
from service.agent.dto.agent_request import AgentRequest
from service.agent.dto.agent_response import AgentResponse
from service.agent.openai_client import APIQueue
from service.agent.openai_service import OpenAIService
from service.retrieval.retrieval_service import RetrievalService

logger = logging.getLogger("rag_agent")


async def evaluate() -> None:
    # Setup
    collections = setup_database()
    retrieval_service = RetrievalService(collections)
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # Start API queue
    api_queue = APIQueue(client)
    await api_queue.start()
    agent_service = OpenAIService(api_queue, retrieval_service)

    # Load dev.csv
    rows: list[dict[str, str]] = []
    with open("data/dev.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    start_time = time.time()

    async def run(idx: int, _row: dict[str, str]) -> bool:
        query: str = (
            f"{_row['question']}\n"
            f"A) {_row['A']}\n"
            f"B) {_row['B']}\n"
            f"C) {_row['C']}\n"
            f"D) {_row['D']}"
        )
        expected = ANSWER_MAP[_row["answer"]]
        request = AgentRequest(query=query)

        response: AgentResponse = await agent_service.answer(
            request, question_id=idx + 1
        )

        is_correct = response.answer == expected
        logger.info(
            f"[{idx + 1}/{total}] expected={expected} predicted={response.answer} "
            f"{'CORRECT' if is_correct else 'WRONG'}"
        )
        return is_correct

    tasks: list[Any] = [run(idx, row) for idx, row in enumerate(rows)]
    results: list[Any] = list(await asyncio.gather(*tasks))
    correct: int = sum(results)

    elapsed: float = time.time() - start_time
    accuracy: float = correct / total
    logger.info(f"accuracy: {correct}/{total} = {accuracy:.4f} ({elapsed:.1f}s)")

    await api_queue.stop()


if __name__ == "__main__":
    import os

    load_dotenv()
    logger.setLevel(logging.DEBUG)
    os.makedirs("./logs", exist_ok=True)
    file_handler = logging.FileHandler(
        "./logs/eval-dev.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(
        StdlibFormatter(
            exclude_fields=["ecs.version", "process", "log.origin", "original"],
            ensure_ascii=False,
        )
    )

    handler = logging.StreamHandler()
    handler.setFormatter(
        StdlibFormatter(
            exclude_fields=["ecs.version", "process", "log.origin", "original"],
            ensure_ascii=False,
        )
    )
    logger.addHandler(handler)
    logger.addHandler(file_handler)
    asyncio.run(evaluate())
