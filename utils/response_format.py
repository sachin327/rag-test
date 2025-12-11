from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from logger import get_logger

logger = get_logger(__name__)


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
    description: str = Field(..., description="Description of the topic")


class SummaryResponse(BaseModel):
    summary: str = Field(..., description="A concise summary of the content")
    topics: List[Topic] = Field(
        default_factory=list,
        description="List of key topics extracted, not more than 10",
    )
    importance_score: float = Field(..., description="Overall importance score (0-1)")


class QuestionItem(BaseModel):
    question_text: str = Field(
        ...,
        description="The question text, do not include option in question if type is MCQ",
    )
    answer: str = Field(..., description="The answer to the question")
    difficulty: str = Field(..., description="Difficulty level (easy, medium, hard)")
    type: str = Field(..., description="Question type (mcq, subjective)")
    topic_keys: List[str] = Field(
        default_factory=list,
        description="List of topic keys associated with the question",
    )
    options: List[str] = Field(
        default_factory=list,
        description="List of options for the question (only for mcq)",
    )
    correct_option_index: int = Field(
        default=0,
        description="Index of the correct option (only for mcq)",
    )


class QuestionResponse(BaseModel):
    questions: List[QuestionItem] = Field(
        default_factory=list,
        description="List of questions generated",
    )


class SourceItem(BaseModel):
    chapter_ids: List[str] = Field(
        ...,
        description="The chapter ids from where answer is generated, take from given context",
    )
    subject_id: str = Field(
        ...,
        description="The subject id from where answer is generated, take from given context",
    )
    class_id: str = Field(
        ...,
        description="The class id from where answer is generated, take from given context",
    )
    source_files: List[str] = Field(
        ...,
        description="The source files id from where answer is generated, take from given context",
    )


class RagQueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="List of source answer generated from",
    )


if __name__ == "__main__":
    response_schema = ResponseSchema(
        json_schema=JsonSchema(
            name="summary",
            schema=SummaryResponse.model_json_schema(),
        )
    )
    logger.info(response_schema.model_dump())
