"""Schema for question generation."""

from typing import List, Optional

from pydantic import BaseModel


# Pydantic models for question generation
class GenerateQuestionsRequest(BaseModel):
    """Request model for question generation."""

    class_id: str
    subject_id: str
    chapter_id: str
    topics: List[str]
    n: int = 10
    mode: str = "or"  # 'or' or 'and'


class QuestionItem(BaseModel):
    """Question item model for question generation."""

    question_text: str
    answer: str
    difficulty: str
    type: str
    topic_keys: List[str]
    source_chunks: List[int]
    options: Optional[List[str]] = None
    correct_option_index: Optional[int] = None


class GenerateQuestionsResponse(BaseModel):
    """Response model for question generation."""

    questions: List[dict]
    count: int
    cached: bool


class IngestDocumentRequest(BaseModel):
    """Request model for document ingestion."""

    class_id: str
    chapter_id: str


class QueryRequest(BaseModel):
    """Request model for document query."""

    query: str
    limit: Optional[int] = 5
    class_id: Optional[str] = None
    chapter_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for document query."""

    answer: Optional[str] = None
    results: List[dict]
    count: int


class UploadResponse(BaseModel):
    """Response model for document upload."""

    filename: str
    message: str
    chunks_added: int
