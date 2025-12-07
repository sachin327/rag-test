from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


class JsonSchema(BaseModel):
    name: str
    strict: bool = True
    schema: Dict[str, Any]


class ResponseSchema(BaseModel):
    type: Literal["json_schema"] = "json_schema"
    json_schema: JsonSchema


class Topic(BaseModel):
    name: str = Field(..., description="Name of the topic, short and concise")
    relevance: float = Field(..., description="Relevance score (0-1)")


class SummaryResponse(BaseModel):
    summary: str = Field(..., description="A concise summary of the content")
    topics: List[Topic] = Field(
        default_factory=list,
        description="List of key topics extracted, not more than 6",
    )
    importance_score: float = Field(..., description="Overall importance score (0-1)")


class RAGResponse(BaseModel):
    answer: str = Field(..., description="The answer to the user's question")
    sources: List[str] = Field(
        default_factory=list, description="List of source files used"
    )
    confidence: float = Field(..., description="Confidence score (0-1)")


if __name__ == "__main__":
    response_schema = ResponseSchema(
        json_schema=JsonSchema(
            name="summary",
            schema=SummaryResponse.model_json_schema(),
        )
    )
    print(response_schema.model_dump())
