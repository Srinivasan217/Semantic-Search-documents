# Task 4 – FastAPI Semantic Search API

## Overview

This task builds a REST API for searching documents using semantic embeddings and Qdrant vector storage.

The API supports:
- Searching documents using natural-language queries.
- Uploading PDF, DOCX, and TXT documents.
- API key authentication.
- Filtering search results by document type.
- Filtering results using a minimum similarity score.
- Evaluating search quality using Hit@1, Hit@3, Hit@5, and MRR.

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Sentence Transformers
- Qdrant
- PyMuPDF
- python-docx
- Pytest

## Project Files

- `main.py` – FastAPI application and endpoints.
- `evaluate_search.py` – Search evaluation script.
- `evaluation_questions.json` – Dataset containing 15 evaluation questions.
- `evaluation_results.json` – Evaluation results.
- `test_api.py` – Six API tests.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API and database status |
| POST | `/search` | Searches for relevant document chunks |
| POST | `/documents` | Uploads and indexes documents |

The search and upload endpoints require an API key.

### Authentication

Pass the API key in the request header:

`x-api-key: training-secret-key`

The application uses `training-secret-key` as its development default. For production, configure a private API key through an environment variable and do not commit secrets to Git.

### Search Example

Request:

```json
{
  "query": "How do employees reset their passwords?",
  "top_k": 5,
  "doc_type": "txt",
  "min_score": 0.5
}
```

The API returns matching document chunks with their source, page, section, and similarity score.

Supported document types: `txt`, `pdf`, and `docx`.

## Search Evaluation

The evaluation uses 15 questions with expected document sources and page numbers.

Two configurations were compared:

1. Baseline Search – semantic search without document-type filtering.
2. Filtered Search – semantic search with document-type filtering.

### Evaluation Results

| Metric | Baseline | Filtered |
|---|---:|---:|
| Hit@1 | 73.33% | 86.67% |
| Hit@3 | 73.33% | 86.67% |
| Hit@5 | 100% | 100% |
| MRR | 0.7933 | 0.9000 |
| Average search time | 0.0043 sec | 0.0034 sec |

Filtered search achieved better Hit@1 and MRR in this evaluation.

## API Testing

Six tests were executed using Pytest.

Test coverage:
1. Health endpoint.
2. Search without an API key.
3. Search with an invalid API key.
4. Search with a valid API key.
5. Search with a document-type filter.
6. Rejection of unsupported upload files.

**Test result: 6 passed, 13 warnings.**

The warnings are deprecation warnings from an installed dependency.

## How to Run

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Start the API from the project root:

```powershell
py -m uvicorn task4_api.main:app --reload
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

Run the evaluation:

```powershell
py .\task4_api\evaluate_search.py
```

Run the API tests:

```powershell
py -m pytest .\task4_api\test_api.py -v
```

**Important:** The local Qdrant database allows only one process to access its storage at a time. Stop the API server before running the evaluation script or other standalone Qdrant inspection scripts.

## Conclusion

The semantic search API was implemented successfully. It supports document uploads, authenticated search, metadata filtering, and search-quality evaluation.

All six API tests passed. Filtered search achieved an MRR of 0.9000 compared with 0.7933 for baseline search.
