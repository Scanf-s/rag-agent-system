import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from ecs_logging import StdlibFormatter
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI

from data.loader import setup_database
from service.agent.dto.agent_request import AgentRequest
from service.agent.dto.agent_response import AgentResponse
from service.agent.openai_client import APIQueue
from service.agent.openai_service import OpenAIService
from service.retrieval.retrieval_service import RetrievalService

# Load environment variables
load_dotenv()
ENV: str = os.environ.get("ENV", "INFO")
DEBUG: bool = ENV == "DEBUG"


# Setup logger
logger = logging.getLogger("rag_agent")
logger.setLevel(ENV)
handler = logging.StreamHandler()
handler.setFormatter(
    StdlibFormatter(
        exclude_fields=["ecs.version", "process", "log.origin", "original"],
        ensure_ascii=False,
    )
)
logging.getLogger("uvicorn").handlers = [handler]
logging.getLogger("uvicorn.access").handlers = [handler]
logging.getLogger("uvicorn.error").handlers = [handler]
logger.addHandler(handler)


# Context manager
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load collections
    collections = setup_database()
    retrieval_service = RetrievalService(collections)
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Start API queue
    api_queue = APIQueue(client)
    await api_queue.start()

    _app.state.agent_service = OpenAIService(api_queue, retrieval_service)

    logger.info("RAG Agent System ready")
    yield

    await api_queue.stop()
    del _app.state.agent_service


# Setup FastAPI application
app: FastAPI = FastAPI(
    debug=DEBUG, title="RAG Agent System", version="0.0.1", lifespan=lifespan
)


@app.post("/predict")
async def predict(request: AgentRequest) -> AgentResponse:
    try:
        service: OpenAIService = app.state.agent_service
        return await service.answer(request)
    except Exception as e:
        logger.error("inference failed: %s", e)
        raise HTTPException(
            status_code=500, detail="inference failed due to the server error"
        ) from e
