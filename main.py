"""Main entry point for the RAG system API."""

import re
import time
import os
import shutil
import asyncio
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Depends
from contextlib import asynccontextmanager

from utils.schema import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QueryRequest,
    QueryResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
)

from services.service import (
    get_upload_service,
    get_generate_question_service,
    get_query_service,
)

from logger import get_logger

# Initialize logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start the keep-alive task
    task = asyncio.create_task(keep_alive())
    yield
    # Shutdown: Cancel the task (optional, but good practice)
    task.cancel()


# Initialize FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="RAG System API",
    description="""
    ## RAG System API

    A Retrieval-Augmented Generation (RAG) system with Qdrant vector database.

    ### Features:
    * Upload documents (TXT, PDF, DOCX)
    * Semantic search using sentence transformers
    * Vector storage with Qdrant

    ### Documentation:
    * **Swagger UI**: [/docs](/docs)
    * **ReDoc**: [/redoc](/redoc)
    """,
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
)


async def keep_alive():
    """Periodically pings the server to keep it active."""
    url = f"{os.getenv('SELF_API_URL')}/ping"  # Adjust URL if deployed elsewhere
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await asyncio.sleep(120)  # Ping every 2 minutes
                logger.info(f"Sending keep-alive ping to {url}")
                response = await client.get(url)
                logger.info(f"Keep-alive ping status: {response.status_code}")
            except Exception as e:
                logger.error(f"Keep-alive ping failed: {e}")


@app.get("/ping")
async def ping():
    return {"status": "alive"}


# Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG System API",
        "documentation": {"swagger_ui": "/docs", "redoc": "/redoc"},
        "endpoints": {
            "POST /upload": "Upload a document (txt, pdf, docx)",
            "POST /query": "Query the RAG system",
            "GET /health": "Health check",
        },
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system for relevant documents and generate an answer."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        service = get_query_service()
        result = service.query(
            query=request.query,
            class_id=request.class_id,
            subject_id=request.subject_id,
            chapter_ids=request.chapter_ids,
        )

        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest):
    """Generate questions for given topics using RAG approach.

    - **class_id**: Document identifier
    - **chapter_id**: Chapter identifier
    - **topics**: List of topic strings
    - **n**: Number of questions to generate (default: 10)
    - **mode**: 'or' (any topic) or 'and' (all topics)
    """
    try:
        service = get_generate_question_service()
        if not service:
            raise HTTPException(
                status_code=503, detail="Question generation service not available"
            )

        questions = service.generate_questions_for_topic_list(
            class_id=request.class_id,
            subject_id=request.subject_id,
            chapter_id=request.chapter_id,
            input_topics=request.topics,
            n=request.n,
            question_type=request.type,
            is_distinct=request.is_distinct,
        )

        return GenerateQuestionsResponse(questions=questions, count=len(questions))
    except Exception as e:
        logger.exception(f"Question generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Question generation failed: {str(e)}"
        )


def document_meta_dependency(
    class_id: str = Form(...),
    chapter_id: str = Form(...),
    subject_id: str = Form(...),
) -> DocumentUploadRequest:
    return DocumentUploadRequest(
        class_id=class_id,
        chapter_id=chapter_id,
        subject_id=subject_id,
    )


@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    meta: DocumentUploadRequest = Depends(document_meta_dependency),
):
    """
    Upload and ingest a document with enhanced metadata extraction.
    Uses multi-stage processing:
    1. Chunks text into 4000 chars and generates summaries
    2. Generates final summary and 5-6 topics from all summaries
    3. Re-chunks to 512 chars with 100 char overlap for storage

    - **file**: Document file (PDF, TXT, DOCX)
    - **class_id**: (Optional) Unique document identifier - auto-generated if not provided
    - **chapter_id**: (Optional) Chapter identifier - auto-generated if not provided
    - **subject_id**: (Optional) Subject identifier - auto-generated if not provided
    """

    # Validate file extension
    allowed_extensions = [".txt", ".pdf", ".docx"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Create uploads directory
    upload_dir = "/tmp/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save file
    file_path = os.path.join(upload_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Process with enhanced ingestion
    try:
        service = get_upload_service()
        if not service:
            raise HTTPException(status_code=503, detail="Upload service not available")

        result = service.upload_document(
            file_path=file_path,
            class_id=meta.class_id,
            chapter_id=meta.chapter_id,
            subject_id=meta.subject_id,
        )

        return DocumentUploadResponse(**result)
    except Exception as e:
        # Clean up file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.exception(f"Enhanced ingestion failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {str(e)}"
        )
    finally:
        # 3. FINALLY block is executed BEFORE the return happens
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
    }
