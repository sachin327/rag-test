"""Main entry point for the RAG system API."""
import re
import time
import os
import shutil
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from schema import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QueryRequest,
    QueryResponse,
)

from services import (
    get_upload_service,
    get_llm_service,
    get_question_service,
    get_rag_system,
)

from logger import get_logger
# Initialize logger
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("=" * 60)
    logger.info("🚀 RAG System API Starting...")
    logger.info("=" * 60)
    logger.info("📚 Swagger UI: http://localhost:8000/docs")
    logger.info("📖 ReDoc: http://localhost:8000/redoc")
    logger.info("🏠 Root: http://localhost:8000/")
    logger.info("=" * 60)
    logger.warning("⚠️  Note: Qdrant connection will be established on first API call")
    logger.warning("   Make sure Qdrant is running on localhost:6333")
    logger.info("   pip install fastembed")
    logger.info("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("=" * 60)
    logger.info("🛑 RAG System API Stopping...")

    global RAG_SYSTEM
    if RAG_SYSTEM:
        logger.info("Closing RAG system connections...")
        RAG_SYSTEM.close()
        logger.info("RAG system connections closed.")

    logger.info("=" * 60)

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
        rag = get_rag_system()
        results = rag.search(
            request.query,
            limit=request.limit,
            class_id=request.class_id,
            chapter_id=request.chapter_id,
        )

        # Generate answer using LLM
        answer = None
        llm = get_llm_service()
        if llm:
            # Use streaming method to print to console while generating
            answer = llm.generate_response_stream(request.query, results)

        return QueryResponse(answer=answer, results=results, count=len(results))
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
        service = get_question_service()
        if not service:
            raise HTTPException(
                status_code=503, detail="Question generation service not available"
            )

        questions = service.generate_questions_for_topic_list(
            class_id=request.class_id,
            chapter_id=request.chapter_id,
            input_topics=request.topics,
            n=request.n,
            mode=request.mode,
        )

        # Check if results are from cache (heuristic: very fast response)
        cached = len(questions) > 0 and all(
            q.get("origin") != "generated" for q in questions
        )

        return GenerateQuestionsResponse(
            questions=questions, count=len(questions), cached=cached
        )
    except Exception as e:
        logger.exception(f"Question generation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Question generation failed: {str(e)}"
        )

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    class_id: Optional[str] = Form(None),
    chapter_id: Optional[str] = Form(None),
    subject_id: Optional[str] = Form(None),
    chapter_name: str = Form(...),
    class_name: str = Form(...),
    subject_name: str = Form(...),
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
    - **chapter_name**: Name of the chapter
    - **class_name**: Name of the class (e.g., "Class 10")
    - **subject_name**: Name of the subject (e.g., "Physics")
    """

    # Validate file extension
    allowed_extensions = [".txt", ".pdf", ".docx"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Generate placeholder IDs if not provided
    timestamp = int(time.time())
    sanitized_filename = re.sub(
        r"[^a-zA-Z0-9]", "_", os.path.splitext(file.filename)[0]
    ).lower()

    if not class_id:
        class_id = f"doc_{sanitized_filename}_{timestamp}"
        logger.info(f"Auto-generated class_id: {class_id}")

    if not chapter_id:
        # Sanitize chapter name for ID
        sanitized_chapter = re.sub(r"[^a-zA-Z0-9]", "_", chapter_name).lower()
        chapter_id = f"ch_{sanitized_chapter}_{timestamp}"
        logger.info(f"Auto-generated chapter_id: {chapter_id}")

    if not subject_id:
        # Sanitize subject name for ID
        sanitized_subject = re.sub(r"[^a-zA-Z0-9]", "_", subject_name).lower()
        subject_id = f"subj_{sanitized_subject}_{timestamp}"
        logger.info(f"Auto-generated subject_id: {subject_id}")

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
            raise HTTPException(
                status_code=503, detail="Upload service not available"
            )

        result = service.upload_document(
            file_path=file_path,
            class_id=class_id,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            class_name=class_name,
            subject_name=subject_name,
            subject_id=subject_id,
        )

        return result
    except Exception as e:
        # Clean up file if processing fails
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.exception(f"Enhanced ingestion failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process document: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
    }
