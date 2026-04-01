from typing import Literal

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    answer: Literal["A", "B", "C", "D"] = Field(
        ..., description="The letter of the correct option"
    )
