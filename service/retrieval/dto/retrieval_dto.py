from pydantic import BaseModel


class Question(BaseModel):
    document: str
    A: str
    B: str
    C: str
    D: str
    answer: str
    category: str
    distance: float


class Retrieval(BaseModel):
    questions: list[Question]
