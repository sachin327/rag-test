# RAG Question Generation System

A comprehensive question generation system that uses Retrieval-Augmented Generation (RAG) with Qdrant vector database, MongoDB for caching, and LLM-powered question generation.

## Features

- **Enhanced Document Ingestion**: Automatically extracts topics, summaries, and importance scores from documents
- **Smart Question Generation**: Uses clustering and LLM to generate diverse, high-quality questions
- **MongoDB Caching**: Caches generated questions for fast retrieval
- **Topic-Based Filtering**: Supports OR/AND filtering by topics
- **Deduplication**: Automatically removes duplicate questions using semantic similarity
- **Redis Streaming**: Real-time streaming of LLM responses via Redis pub/sub

## Architecture

### Components

1. **Ingestion Service** (`ingestion_service.py`)
   - Chunks documents with sentence-aware splitting
   - Extracts topics and summaries using LLM
   - Computes importance scores
   - Stores in Qdrant with rich metadata

2. **Question Generation Service** (`question_generation_service.py`)
   - Queries Qdrant with topic filters
   - Clusters candidates for coverage
   - Generates questions via LLM
   - Deduplicates and selects diverse questions
   - Caches in MongoDB

3. **Database Layers**
   - **Qdrant** (`qdrant_db.py`): Vector storage for document chunks
   - **MongoDB** (`mongo_db.py`): Question cache and metadata
   - **Redis** (`redis_db.py`): Real-time streaming and pub/sub

4. **Utilities** (`question_utils.py`)
   - Topic normalization
   - Embedding clustering
   - Similarity-based deduplication
   - Question selection algorithms

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start required services
docker run -p 6333:6333 qdrant/qdrant  # Qdrant
docker run -p 27017:27017 mongo        # MongoDB
docker run -p 6379:6379 redis          # Redis
```

## Configuration

Update `.env` file with your settings:

```env
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=rag_db

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_CHANNEL=mychannel

# API Keys
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

## API Endpoints

### 1. Enhanced Document Ingestion

```bash
POST /ingest-enhanced
```

Upload a document with automatic topic extraction and metadata enrichment.

**Request:**
```bash
curl -X POST "http://localhost:8000/ingest-enhanced" \
  -F "file=@document.pdf" \
  -F "class_id=9th_ncert" \
  -F "chapter_id=chapter_1"
```

**Response:**
```json
{
  "success": true,
  "class_id": "9th_ncert",
  "chapter_id": "chapter_1",
  "chunks_processed": 45,
  "elapsed_time": 12.34
}
```

### 2. Generate Questions

```bash
POST /generate-questions
```

Generate questions for specific topics using RAG.

**Request:**
```json
{
  "class_id": "9th_ncert",
  "chapter_id": "chapter_1",
  "topics": ["fluids", "pressure", "states_of_matter"],
  "n": 10,
  "mode": "or"
}
```

**Response:**
```json
{
  "questions": [
    {
      "question_text": "Why do liquids take the shape of their container?",
      "answer": "Because liquid molecules can move freely...",
      "difficulty": "easy",
      "type": "conceptual",
      "topic_keys": ["fluids", "states_of_matter"],
      "source_chunks": [39, 40],
      "_id": "507f1f77bcf86cd799439011"
    }
  ],
  "count": 10,
  "cached": false
}
```

### 3. Query Cached Questions

```bash
GET /questions?class_id=9th_ncert&chapter_id=chapter_1&topics=fluids&topics=pressure&limit=20&mode=or
```

Retrieve previously generated questions from cache.

### 4. Standard RAG Query

```bash
POST /query
```

Query the RAG system for document retrieval and answer generation.

## Data Schemas

### Qdrant Chunk Schema

```json
{
  "class_id": "9th_ncert",
  "chapter_id": "chapter_1",
  "chunk_index": 39,
  "text": "...",
  "topic_keys": ["fluids", "states_of_matter"],
  "summary": "One-sentence summary",
  "is_heading": false,
  "importance_score": 0.75,
  "token_count": 210,
  "sentence_count": 5,
  "source_file": "physics.pdf",
  "created_at": 1764269614.91
}
```

### MongoDB Question Schema

```json
{
  "_id": "ObjectId",
  "class_id": "9th_ncert",
  "chapter_id": "chapter_1",
  "question_text": "...",
  "answer": "...",
  "topic_keys": ["fluids", "pressure"],
  "source_chunks": [12, 13],
  "difficulty": "easy|medium|hard",
  "type": "fact|conceptual|mcq|short_answer",
  "created_at": 1764269614.91,
  "status": "draft|published",
  "origin": "generated|human_edited",
  "embedding": [...],
  "overlap_count": 2,
  "options": ["A", "B", "C", "D"],
  "correct_option_index": 0
}
```

## Usage Examples

### Python Client

```python
import requests

# 1. Ingest a document
with open('textbook.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ingest-enhanced',
        files={'file': f},
        data={
            'class_id': '9th_ncert',
            'chapter_id': 'chapter_1'
        }
    )
print(response.json())

# 2. Generate questions
response = requests.post(
    'http://localhost:8000/generate-questions',
    json={
        'class_id': '9th_ncert',
        'chapter_id': 'chapter_1',
        'topics': ['fluids', 'pressure'],
        'n': 5,
        'mode': 'or'
    }
)
questions = response.json()['questions']

# 3. Query cached questions
response = requests.get(
    'http://localhost:8000/questions',
    params={
        'class_id': '9th_ncert',
        'chapter_id': 'chapter_1',
        'topics': ['fluids'],
        'limit': 10
    }
)
cached = response.json()['questions']
```

### Subscribe to Redis Stream

```python
from redis_db import RedisDB

def handle_stream(message):
    print(f"Response: {message['response']}")
    if message['finish_reason']:
        print(f"Stream ended: {message['finish_reason']}")

redis = RedisDB()
redis.subscribe(handle_stream)  # Blocks and listens
```

## Configuration Parameters

### Question Generation Service

```python
service = QuestionGenerationService()

# Configurable parameters
service.candidate_pool_size = 80          # Max chunks to retrieve
service.token_budget_per_cluster = 1200   # Max tokens per cluster
service.dedupe_threshold = 0.92           # Similarity threshold
service.questions_per_cluster = 2         # Questions per cluster
```

### Topic Normalization

Topics are automatically normalized:
- Converted to lowercase
- Spaces replaced with underscores
- Punctuation removed
- Aliases applied (configurable in `question_utils.py`)

Example:
- "States of Matter" → "states_of_matter"
- "Fluid Mechanics" → "fluids" (via alias)

## Testing

```bash
# Test MongoDB connection
python mongo_db.py

# Test Redis connection
python redis_db.py

# Test question generation
python question_generation_service.py

# Test enhanced ingestion
python ingestion_service.py
```

## Performance Optimization

1. **Caching**: MongoDB caches generated questions for instant retrieval
2. **Clustering**: Reduces LLM calls by grouping similar chunks
3. **Lazy Loading**: Models loaded only when needed
4. **Batch Processing**: Processes multiple chunks in single LLM call
5. **FastEmbed**: Optional fast embedding library (50x faster)

## Monitoring

Check service health:

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "rag_initialized": true,
  "question_service_initialized": true
}
```

## Troubleshooting

### MongoDB Connection Issues

```bash
# Check if MongoDB is running
docker ps | grep mongo

# Test connection
python -c "from mongo_db import MongoDB; m = MongoDB(); print('OK')"
```

### Qdrant Connection Issues

```bash
# Check Qdrant
curl http://localhost:6333/collections

# Test connection
python -c "from qdrant_db import QdrantDB; q = QdrantDB(); print('OK')"
```

### Redis Connection Issues

```bash
# Check Redis
redis-cli ping

# Test connection
python -c "from redis_db import RedisDB; r = RedisDB(); print('OK')"
```

## Advanced Features

### Custom Topic Aliases

Edit `question_utils.py`:

```python
TOPIC_ALIASES = {
    "states of matter": "states_of_matter",
    "fluid mechanics": "fluids",
    "your_alias": "canonical_name"
}
```

### Custom Question Types

Modify the LLM prompt in `question_generation_service.py` to support custom question types.

### Importance Scoring

Customize importance scoring in `ingestion_service.py`:

```python
def compute_importance_score(self, text: str, is_heading: bool = False) -> float:
    # Your custom logic here
    pass
```

## License

MIT

## Support

For issues and questions, please check the logs:
- Application logs: Check console output
- MongoDB logs: `docker logs <mongo_container>`
- Qdrant logs: `docker logs <qdrant_container>`
