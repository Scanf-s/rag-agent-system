import asyncio
import csv
import logging
import os
import sys
import time
from collections.abc import Coroutine
from typing import Any

import httpx
from ecs_logging import StdlibFormatter

from data.loader import ANSWER_MAP

logger = logging.getLogger("rag_agent")

API_URL = "http://localhost:8000/predict"
CONCURRENCY = 5


async def evaluate(csv_path: str) -> None:
    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    start_time = time.time()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def run(_client: httpx.AsyncClient, idx: int, _row: dict[str, str]) -> bool:
        query = (
            f"{_row['question']}\n"
            f"A) {_row['A']}\n"
            f"B) {_row['B']}\n"
            f"C) {_row['C']}\n"
            f"D) {_row['D']}"
        )
        expected = ANSWER_MAP[_row["answer"]]

        async with semaphore:
            resp: Any = await _client.post(API_URL, json={"query": query})
            resp.raise_for_status()
            result: dict[str, Any] = resp.json()

        predicted: str = result["answer"]
        is_correct: bool = predicted == expected
        logger.info(
            f"[{idx + 1}/{total}] expected={expected} predicted={predicted} "
            f"{'CORRECT' if is_correct else 'WRONG'}"
        )
        return is_correct

    async with httpx.AsyncClient(timeout=180) as client:
        tasks: list[Coroutine[Any, Any, bool]] = [
            run(client, idx, row) for idx, row in enumerate(rows)
        ]
        results: list[Any] = list(await asyncio.gather(*tasks))

    correct: int = sum(results)
    elapsed: float = time.time() - start_time
    accuracy: float = correct / total
    logger.info(f"accuracy: {correct}/{total} = {accuracy:.4f} ({elapsed:.1f}s)")


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    os.makedirs("./logs", exist_ok=True)
    file_handler = logging.FileHandler("./logs/eval.log", mode="w", encoding="utf-8")
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
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/test.csv"

    try:
        asyncio.run(evaluate(csv_file))
    except Exception as e:
        logger.error(f"evaluation failed: {e}")
