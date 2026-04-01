import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI, BadRequestError, RateLimitError

from config import MAX_RETRIES

logger = logging.getLogger("rag_agent")

# Default number of concurrent API workers
DEFAULT_WORKERS = 5


class APIQueue:
    def __init__(self, client: AsyncOpenAI, max_workers: int = DEFAULT_WORKERS):
        self.client: AsyncOpenAI = client
        self._queue: asyncio.Queue[tuple[dict[str, Any], asyncio.Future[Any], int]] = (
            asyncio.Queue()
        )
        self._workers: list[asyncio.Task[None]] = []
        self._max_workers: int = max_workers

    async def start(self) -> None:
        for _ in range(self._max_workers):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)
        logger.info("APIQueue started with %d workers", self._max_workers)

    async def stop(self) -> None:
        for _ in self._workers:
            future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
            future.set_result(None)
            await self._queue.put(({}, future, -1))

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("APIQueue stopped")

    async def submit(self, **kwargs: Any) -> Any:
        # Create a new future container that receives the coroutine result
        future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()

        # Put a task into the queue. Worker coroutine will pull this task automatically.
        await self._queue.put((kwargs, future, 0))

        # Wait coroutine returns a result into the future and return it.
        return await future

    async def _worker(self) -> None:
        while True:
            kwargs, future, attempt = await self._queue.get()

            # Shutdown sentinel
            if attempt < 0 and future.done():
                self._queue.task_done()
                break

            if future.cancelled():
                self._queue.task_done()
                continue

            try:
                result = await self.client.chat.completions.create(**kwargs)  # pyright: ignore[reportUnknownVariableType]
                future.set_result(result)
            except (RateLimitError, BadRequestError):
                # If the rate limit error or bad request error occurred,
                # sleep for a while by the exponential backoff strategy,
                # and re-queue the task if the attempt is within the MAX_RETRIES
                if attempt < MAX_RETRIES:
                    wait: int = 2 ** (attempt + 1)
                    logger.info(
                        "rate limit (bad request) occurred, attempt %d/%d, wait %ds",
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    await self._queue.put((kwargs, future, attempt + 1))
                else:
                    logger.warning(
                        "max retries reached due to rate limit or bad request occurred in the last attempt"
                    )
                    future.set_result(None)
            except Exception as e:
                logger.error("unexpected error: %s", e)
                future.set_result(None)
            finally:
                self._queue.task_done()
