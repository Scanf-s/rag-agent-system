from typing import Any

from pydantic import BaseModel


class Collections(BaseModel):
    questions: Any
