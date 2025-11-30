# RAG System API

A Retrieval-Augmented Generation (RAG) system with FastAPI and Qdrant vector database.

## Prerequisites

1. **Python 3.8+**
2. **Qdrant** running on `localhost:6333`

## Installation

Install required dependencies:

```bash
pip install fastapi uvicorn qdrant-client sentence-transformers pypdf python-docx torch
```

## Running the Application

### Step 1: Start Qdrant

You need Qdrant running before using the upload/query endpoints. You have two options:

**Option A: Using Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option B: Using Docker Compose**
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Then run:
```bash
docker-compose up -d
```

### Step 2: Start FastAPI Server

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Or for external access:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Accessing the API

Once the server is running, you can access:

- **Swagger UI (Interactive Docs)**: http://localhost:8000/docs
- **ReDoc (Alternative Docs)**: http://localhost:8000/redoc
- **Root Endpoint**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health

## API Endpoints

### 1. Upload Document
**POST** `/upload`

Upload a document (TXT, PDF, or DOCX) to the RAG system.

### 2. Query Documents
**POST** `/query`

Search for relevant document chunks using semantic search.

Example request body:
```json
{
  "query": "What is machine learning?",
  "limit": 5
}
```

### 3. Health Check
**GET** `/health`

Check if the API is running and if RAG system is initialized.

## Troubleshooting

### Server won't start
- Make sure all dependencies are installed
- Check if port 8000 is available

### Can't see Swagger UI
- Ensure the server is running (check console output)
- Navigate to http://localhost:8000/docs
- Try clearing browser cache or use incognito mode

### Upload/Query endpoints fail
- Make sure Qdrant is running on `localhost:6333`
- Check Qdrant logs for errors
- Verify Qdrant is accessible: http://localhost:6333/dashboard

### "Failed to connect to Qdrant" error
- Start Qdrant using Docker (see Step 1 above)
- Verify Qdrant is running: `docker ps` or check http://localhost:6333

## Project Structure

```
RAG/
├── main.py              # FastAPI application
├── rag.py               # RAG system logic
├── qdrant_db.py         # Qdrant database wrapper
├── document_loader.py   # Document loading utilities
├── uploads/             # Uploaded documents (created automatically)
└── README.md            # This file
```

## Notes

- The server will start successfully even if Qdrant is not running
- Qdrant connection is only established when you use upload/query endpoints
- Uploaded files are stored in the `uploads/` directory
- The RAG system uses the `all-MiniLM-L6-v2` sentence transformer model
