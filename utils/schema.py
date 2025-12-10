"""Schema for question generation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


# Pydantic models for question generation
class GenerateQuestionsRequest(BaseModel):
    """Request model for question generation."""

    class_id: str
    subject_id: str
    chapter_id: str
    topics: List[str]
    n: int = 10
    type: str = "mcq"  # 'mcq' or 'subjective'


class QuestionItem(BaseModel):
    """Question item model for question generation."""

    question_text: str
    answer: str
    difficulty: str
    type: str
    topic_keys: List[str]
    options: Optional[List[str]] = None
    correct_option_index: Optional[int] = None


class GenerateQuestionsResponse(BaseModel):
    """Response model for question generation."""

    questions: List[QuestionItem]
    count: int


class DocumentUploadRequest(BaseModel):
    class_id: str
    chapter_id: str
    subject_id: str
    class_name: str
    chapter_name: str
    subject_name: str


class DocumentUploadResponse(BaseModel):
    success: bool
    metadata: DocumentUploadRequest
    topics_extracted: int
    topic_keys: List[Dict[str, Any]]
    summary: str
    summary_length: int
