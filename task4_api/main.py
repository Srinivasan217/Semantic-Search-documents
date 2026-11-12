
import os
import re
import io
import uuid
from pathlib import Path
from datetime import date
from contextlib import asynccontextmanager

import pymupdf
from docx import Document
from sentence_transformers import SentenceTransformer

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Depends,
    Header,
    Request,
)
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DB_PATH = BASE_DIR / "task3_vector_storage" / "output" / "qdrant_db"

COLLECTION_NAME = "document_chunks"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_SIZE = 384

API_KEY = os.getenv("API_KEY", "training-secret-key")

MAX_FILE_SIZE = 10 * 1024 * 1024


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def verify_api_key(x_api_key: str = Header(default=None)):

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class SearchRequest(BaseModel):

    query: str = Field(min_length=1)

    top_k: int = Field(default=5, ge=1, le=20)

    doc_type: str | None = Field(
        default=None,
        pattern="^(txt|pdf|docx)$",
    )

    min_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )


# --------------------------------------------------
# APPLICATION LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading embedding model...")

    app.state.model = SentenceTransformer(MODEL_NAME)

    print("Connecting to Qdrant...")

    app.state.client = QdrantClient(path=str(DB_PATH))

    collections = [
        item.name
        for item in app.state.client.get_collections().collections
    ]

    if COLLECTION_NAME not in collections:

        app.state.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

    print("FastAPI application is ready!")

    yield

    app.state.client.close()

    print("Qdrant connection closed.")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Semantic Document Search API",
    description="Search and manage documents using semantic embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check(request: Request):

    try:

        request.app.state.client.get_collection(COLLECTION_NAME)

        return {
            "status": "healthy",
            "database": "connected",
            "collection": COLLECTION_NAME,
        }

    except Exception:

        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable",
        )


# --------------------------------------------------
# SEMANTIC SEARCH WITH FILTERS
# --------------------------------------------------

@app.post("/search", dependencies=[Depends(verify_api_key)])
def search_documents(
    data: SearchRequest,
    request: Request,
):

    model = request.app.state.model
    client = request.app.state.client

    query_vector = model.encode(
        data.query,
        normalize_embeddings=True,
    ).tolist()

    # Document type filter
    query_filter = None

    if data.doc_type:

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_type",
                    match=models.MatchValue(
                        value=data.doc_type
                    ),
                )
            ]
        )

    # Minimum similarity score
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=data.top_k,
        query_filter=query_filter,
        score_threshold=data.min_score,
        with_payload=True,
    ).points

    search_results = []

    for result in results:

        payload = result.payload or {}

        search_results.append({
            "id": str(result.id),
            "score": round(float(result.score), 4),
            "text": payload.get("text", ""),
            "source": payload.get("source", ""),
            "page": payload.get("page", 1),
            "section_heading": payload.get("section_heading", ""),
            "doc_type": payload.get("doc_type", ""),
        })

    return {
        "query": data.query,
        "filters": {
            "doc_type": data.doc_type,
            "min_score": data.min_score,
        },
        "total_results": len(search_results),
        "results": search_results,
    }


# --------------------------------------------------
# DOCUMENT EXTRACTION
# --------------------------------------------------

def extract_document(content: bytes, extension: str):

    pages = []

    if extension == ".txt":

        text = content.decode("utf-8")

        pages.append((1, text))

    elif extension == ".pdf":

        pdf = pymupdf.open(
            stream=content,
            filetype="pdf",
        )

        for page_number, page in enumerate(pdf, start=1):

            pages.append((
                page_number,
                page.get_text(),
            ))

        pdf.close()

    elif extension == ".docx":

        document = Document(io.BytesIO(content))

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        pages.append((1, text))

    else:

        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use TXT, PDF, or DOCX.",
        )

    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(text: str, max_characters: int = 500):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_characters:

            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            for start in range(0, len(sentence), max_characters):

                chunks.append(
                    sentence[start:start + max_characters]
                )

            continue

        if (
            current_chunk
            and len(current_chunk) + len(sentence) + 1 > max_characters
        ):

            chunks.append(current_chunk)
            current_chunk = sentence

        else:

            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# --------------------------------------------------
# DOCUMENT UPLOAD AND INDEXING
# --------------------------------------------------

@app.post("/documents", dependencies=[Depends(verify_api_key)])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):

    filename = Path(file.filename or "").name

    extension = Path(filename).suffix.lower()

    if extension not in [".txt", ".pdf", ".docx"]:

        raise HTTPException(
            status_code=400,
            detail="Only TXT, PDF, and DOCX files are supported.",
        )

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail="File size must not exceed 10 MB.",
        )

    pages = extract_document(content, extension)

    model = request.app.state.model
    client = request.app.state.client

    points = []
    total_chunks = 0

    for page_number, page_text in pages:

        chunks = create_chunks(page_text)

        if not chunks:
            continue

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
        )

        for chunk_index, (chunk, vector) in enumerate(
            zip(chunks, embeddings),
            start=1,
        ):

            point_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{filename}|{page_number}|{chunk_index}|{chunk}",
            )

            points.append(
                models.PointStruct(
                    id=str(point_id),
                    vector=vector.tolist(),
                    payload={
                        "text": chunk,
                        "source": filename,
                        "page": page_number,
                        "section_heading": chunk[:120],
                        "created_date": date.today().isoformat(),
                        "chunk_index": chunk_index,
                        "doc_type": extension.lstrip("."),
                        "method": "sentence_based",
                    },
                )
            )

            total_chunks += 1

    if not points:

        raise HTTPException(
            status_code=400,
            detail="No readable text found in the document.",
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return {
        "message": "Document indexed successfully",
        "filename": filename,
        "total_chunks": total_chunks,
    }
