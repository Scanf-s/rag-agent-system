import asyncio
import json
import logging
from typing import Any

from config import MODEL, SEED, TEMPERATURE
from service.agent.dto.agent_request import AgentRequest
from service.agent.dto.agent_response import AgentResponse
from service.agent.openai_client import APIQueue
from service.agent.prompt.response_schema import RESPONSE_SCHEMA
from service.agent.prompt.system_prompt import SYSTEM_PROMPT
from service.retrieval.dto.retrieval_dto import Retrieval
from service.retrieval.retrieval_service import RetrievalService

logger = logging.getLogger("rag_agent")


class OpenAIService:
    def __init__(self, api_queue: APIQueue, retrieval_service: RetrievalService):
        self.api_queue: APIQueue = api_queue
        self.retrieval_service: RetrievalService = retrieval_service

    async def answer(
        self, request: AgentRequest, question_id: int | None = None
    ) -> AgentResponse:
        extra: dict[str, Any] = (
            {"question.id": question_id} if question_id is not None else {}
        )

        # Retrieve related data from vector db
        retrieval: Retrieval = await asyncio.to_thread(
            self.retrieval_service.search, request.query, extra
        )

        # Build prompt
        messages = self._build_messages(query=request.query, retrieval=retrieval)

        # Submit to API queue
        resp: Any = await self.api_queue.submit(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            seed=SEED,
            response_format=RESPONSE_SCHEMA,
        )
        if resp is None:
            raise RuntimeError("OpenAI API call failed")

        content: str = resp.choices[0].message.content or "{}"
        try:
            result: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed, content=%s", content[:100], extra=extra)  # pyright: ignore[reportCallIssue]
            return AgentResponse(answer="A")

        logger.debug(
            "reasoning=%s answer=%s",
            result.get("reasoning", "")[:200],
            result.get("answer"),
            extra=extra,  # pyright: ignore[reportCallIssue]
        )

        return AgentResponse(answer=result.get("answer", "A"))

    @staticmethod
    def _build_messages(
        query: str,
        retrieval: Retrieval,
    ) -> list[dict[str, Any]]:
        sections: list[str] = []
        if retrieval.questions:
            question_section = ""
            for i, q in enumerate(retrieval.questions, 1):
                question_section += (
                    f"{i}. [{q.category}] {q.document}\n"
                    f"A) {q.A}\nB) {q.B}\nC) {q.C}\nD) {q.D}\n"
                    f"정답: {q.answer}\n\n"
                )
            sections.append(f"### 유사 기출문제\n{question_section}")

        sections.append(f"### 문제\n{query}")
        user_content = "\n".join(sections)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]


# For debugging purposes
if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    from ecs_logging import StdlibFormatter
    from openai import AsyncOpenAI

    from data.loader import setup_database

    load_dotenv()
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
    service: RetrievalService = RetrievalService(setup_database())
    logger.info("data loading completed.")

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    queue = APIQueue(client)

    async def main() -> None:
        await queue.start()
        agent_service = OpenAIService(queue, service)
        request = AgentRequest(
            query="""형벌 유형 중 재산형에 포함되지 않는 것은?
            A) 금고
            B) 벌금
            C) 과료
            D) 몰수
            """
        )
        response = await agent_service.answer(request)
        logger.debug(response.answer)
        await queue.stop()

    asyncio.run(main())
