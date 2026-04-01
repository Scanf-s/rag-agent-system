from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    query: str = Field(
        ...,
        description="The multiple-choice legal question to be analyzed and answered",
    )
