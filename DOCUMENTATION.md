# AI Service Documentation

This document provides a comprehensive overview of the AI Service, designed for both business stakeholders and technical developers.

---

# Part 1: Executive Summary (For Founders & Stakeholders)

## Product Overview
The **AI Service** is an intelligent backend system designed to transform static educational content (textbooks, PDFs, documents) into an interactive, queryable knowledge base. It leverages advanced Artificial Intelligence (AI) to understand, summarize, and generate assessment content from uploaded materials.

## Key Capabilities

### 1. Intelligent Document Analysis
- **Multi-Format Support:** Accepts PDF, DOCX, and TXT files.
- **Deep Understanding:** Automatically reads and "understands" the content, breaking it down into manageable concepts.
- **Smart Summarization:** Generates concise summaries of chapters and topics, making it easier to digest large volumes of information.

### 2. Semantic Search (RAG - Retrieval Augmented Generation)
- **Natural Language Queries:** Users can ask questions in plain English (e.g., "What is Newton's Second Law?") rather than just keyword matching.
- **Context-Aware Answers:** The system retrieves the exact paragraphs relevant to the question and uses a Large Language Model (LLM) to formulate a precise answer.

### 3. Automated Question Generation
- **Teacher's Aid:** Automatically generates quiz questions (Multiple Choice, Short Answer, etc.) from any chapter or topic.
- **Topic-Specific:** Can generate questions focused on specific concepts (e.g., "Generate 10 hard questions about Thermodynamics").
- **Diversity & Quality:** Uses clustering algorithms to ensure questions cover different aspects of the topic and avoids duplicates.

## Business Value
- **Efficiency:** Drastically reduces the time required to create study materials, quizzes, and summaries.
- **Scalability:** Can process thousands of documents and serve thousands of students simultaneously.
- **Personalization:** Enables adaptive learning platforms by generating content tailored to specific topics or difficulty levels.

## Use Cases
- **EdTech Platforms:** Powering "Ask a Doubt" features and automated quiz generation.
- **Corporate Training:** Creating instant training modules and assessments from internal manuals.
- **Knowledge Management:** allowing employees to instantly find answers from vast internal documentation.

---

# Part 2: Developer Guide (For Technical Team)

## System Architecture

The system follows a **Retrieval-Augmented Generation (RAG)** architecture:

1.  **Ingestion Layer:**
    - Documents are uploaded via API.
    - **Processing:** Text is extracted and split into "chunks".
    - **Enrichment:** An LLM generates summaries and extracts key topics for each section.
    - **Embedding:** Text chunks are converted into vector embeddings (numerical representations of meaning).

2.  **Storage Layer:**
    - **Qdrant (Vector DB):** Stores embeddings for semantic search.
    - **MongoDB:** Stores generated questions and caching.

3.  **Retrieval & Generation Layer:**
    - **Query:** User asks a question.
    - **Search:** System finds the most relevant chunks in Qdrant.
    - **Generation:** An LLM (Google Gemini) uses the retrieved chunks to generate an answer or questions.

## Technology Stack
- **Language:** Python 3.8+
- **Framework:** FastAPI (High-performance web framework)
- **Vector Database:** Qdrant (running via Docker)
- **LLM Integration:** Google Gemini (via custom `LLMService`)
- **Database:** MongoDB (for structured data)
- **Utilities:** `sentence-transformers` (for embeddings), `pypdf`, `python-docx`

## Setup & Installation

### Prerequisites
- Docker & Docker Compose
- Python 3.8+
- API Keys for Google Gemini (and optionally OpenRouter)

### Environment Variables
Create a `.env` file in the root directory:
```env
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# LLM Configuration
GEMINI_API_KEY=your_gemini_key
```

### Running Locally
1.  **Start Qdrant:**
    ```bash
    docker run -p 6333:6333 qdrant/qdrant
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Start API Server:**
    ```bash
    uvicorn main:app --reload
    ```
    Access Swagger UI at `http://localhost:8000/docs`.

## API Reference

### 1. Upload Document
**Endpoint:** `POST /upload-document`

Ingests a document into the system. This triggers the "Enhanced Ingestion" pipeline.

**Parameters:**
- `file`: The document file (PDF, TXT, DOCX).
- `class_name`, `subject_name`, `chapter_name`: Metadata for organization.
- `class_id`, `subject_id`, `chapter_id`: (Optional) Unique IDs. Auto-generated if omitted.

**Internal Process:**
1.  **Load:** Text is extracted.
2.  **Summarize:** Text is split into large chunks (4000 chars) for summarization and topic extraction.
3.  **Vectorize:** Text is re-split into smaller chunks (512 chars, 100 overlap) for vector search.
4.  **Store:** Vectors and metadata are saved to Qdrant.

### 2. Query System
**Endpoint:** `POST /query`

Ask a question based on the uploaded documents.

**Body:**
```json
{
  "query": "What is the capital of France?",
  "limit": 5
}
```

### 3. Generate Questions
**Endpoint:** `POST /generate-questions`

Generates assessment questions for specific topics.

**Body:**
```json
{
  "class_id": "class_10",
  "chapter_id": "ch_physics_01",
  "topics": ["Newton's Laws", "Gravity"],
  "n": 10,
  "mode": "or"
}
```

**Logic:**
1.  **Retrieve:** Fetches relevant text chunks from Qdrant based on topics.
2.  **Cluster:** Groups chunks to ensure diversity.
3.  **Generate:** Uses LLM to create questions for each cluster.
4.  **Deduplicate:** Removes similar questions using embedding similarity.
5.  **Cache:** Stores results in MongoDB for future fast retrieval.

## Project Structure
- `main.py`: FastAPI entry point and route definitions.
- `services/`: Business logic.
    - `rag.py`: Core RAG implementation (chunking, searching).
    - `upload_service.py`: Orchestrates document ingestion.
    - `question_generation_service.py`: Logic for generating questions.
- `db/`: Database interactions.
    - `qdrant_db.py`: Qdrant client wrapper.
    - `mongo_db.py`: MongoDB client wrapper.
- `llm/`: LLM integration (Gemini, etc.).
