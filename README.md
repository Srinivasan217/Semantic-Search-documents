# Semantic Search Document Search
A python-based semantic document search system that uses embeddings,vector databases, and FastAPI to search documents based on meaning rather than exact keyword matching.
Project Overview

This project explores the complete workflow of building a semantic document search application:

Generate text embeddings.

Calculate semantic similarity.

Split documents into searchable chunks.

Generate stable IDs for document chunks.

Store and retrieve embeddings using vector databases.

Build a REST API using FastAPI.

Evaluate search quality and performance.

Technologies Used

Python – Main programming language

Sentence Transformers – Text embedding generation

NumPy – Numerical operations

Scikit-learn – Similarity calculations

FAISS – Vector similarity search and benchmarking

Qdrant – Vector database

FastAPI – REST API development

Pytest – API testing

PyMuPDF – PDF text extraction

python-docx – DOCX document processing

Project Structure

document-search/
│
├── task1_embeddings/
│   ├── common.py
│   ├── cosine_similarity.py
│   ├── embeddings_comparison.py
│   ├── similarity_matrix.py
│   ├── use_cases.py
│   └── README.md
│
├── task2_chunking/
│   ├── documents/
│   ├── output/
│   ├── chunk_size_experiment.py
│   ├── chunking_methods.py
│   ├── create_documents.py
│   ├── create_experiment_data.py
│   ├── evaluation_questions.py
│   ├── extract_documents.py
│   ├── inspect_corpus.py
│   ├── sentence_boundary_demo.py
│   ├── stable_id_edit_test.py
│   └── stable_ids.py
│
├── task3_vector_storage/
│   ├── output/
│   ├── faiss_benchmark.py
│   ├── qdrant_index_status.py
│   ├── qdrant_pdf_filter.py
│   ├── qdrant_storage.py
│   └── README.md
│
├── task4_api/
│   ├── main.py
│   ├── evaluate_search.py
│   ├── evaluation_questions.json
│   ├── evaluation_results.json
│   ├── test_api.py
│   ├── requirements.txt
│   └── README.md
│
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md

TAsk1 -Embedding and similarity
This task covers the fundamentals of text embeddings and semantic similarity.

Topics covered
Task 1 – Embeddings and Similarity

This task covers the fundamentals of text embeddings and semantic similarity.

Topics covered

Text embedding generation

Sentence Transformers

Embedding comparison

Cosine similarity

Similarity matrices

Semantic search use cases

Cosine Similarity Results

Input vectors

Expected similarity

[1,2,3] and [2,4,6]

1.0

[1,0] and [0,1]

0.0

[1,0] and [-1,0]

-1.0

These results demonstrate identical, perpendicular, and opposite vector directions.

Task 2 – Document Chunking

This task focuses on preparing documents for semantic search.

Topics covered

PDF, TXT and DOCX text extraction

Fixed-size chunking

Sentence-based chunking

Recursive chunking

Chunk size and overlap experiments

Metadata management

Stable chunk ID generation

Stable ID Verification

The edit test produced:

Changed chunk IDs: 3
Unchanged chunk IDs: 244
Original documents were not modified.

This verifies that changed chunks receive new IDs while unchanged chunks retain their existing IDs.

Task 3 – Vector Storage and Search

This task covers storing document embeddings and retrieving similar vectors efficiently.

Topics covered

Vector database fundamentals

Qdrant local storage

Vector indexing

Similarity search

Metadata filtering

PDF-specific filtering

FAISS exact search

HNSW approximate search

Recall@10 benchmarking

FAISS Benchmark Results

The benchmark used:

50,000 vectors

384 dimensions

100 queries

Top 10 search results

Search configuration

Search time

Recall@10

Exact search

0.7128 sec

100% reference

HNSW – 64

0.0453 sec

52.90%

HNSW – 128

0.0962 sec

67.10%

HNSW – 256

0.1341 sec

81.60%

HNSW – 512

0.2064 sec

92.00%

The best tested HNSW configuration achieved 92% Recall@10 with efSearch=512.

Note: Benchmark timings depend on the hardware and runtime environment.

Task 4 – FastAPI Application and Evaluation

This task implements a REST API for semantic document search.

API Features

Health check endpoint

Semantic search endpoint

Document upload and indexing

API key authentication

Document type filtering

Minimum similarity score filtering

PDF, TXT and DOCX document support

Search evaluation

Automated API tests

API Endpoints

Method

Endpoint

Purpose

GET

/health

Check API health

POST

/search

Search indexed documents

POST

/documents

Upload and index documents

Running the API

From the project root, activate the virtual environment:

.\venv\Scripts\Activate.ps1

Move into the API folder:

cd task4_api

Start the FastAPI server:

py -m uvicorn main:app --reload

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Check API health:

http://127.0.0.1:8000/health

API Testing

Run the automated tests from the task4_api folder:

py -m pytest test_api.py -v

Latest verified test result:

6 passed

Search Evaluation

The API evaluation compares baseline semantic search with metadata-filtered semantic search.

Latest 15-question evaluation:

Metric

Baseline

Filtered

Hit@1

0.7333

0.8667

Hit@3

0.7333

0.8667

Hit@5

1.0000

1.0000

MRR

0.7933

0.9000

The filtered search performed better on Hit@1, Hit@3 and MRR in this evaluation.

Note: This comparison measures metadata filtering, not a comparison between two embedding models.

Installation and Setup

Requirements

Python 3.11 or later

Windows, Linux or macOS

pip

Git

1. Clone the repository

git clone https://github.com/Srinivasan217/Semantic-Search-documents.git

2. Move into the project folder

cd Semantic-Search-documents

3. Create a virtual environment

py -m venv venv

4. Activate the environment

.\venv\Scripts\Activate.ps1

5. Install dependencies

py -m pip install -r requirements.txt

6. Run the application

cd task4_api
py -m uvicorn main:app --reload

Running Individual Tasks

Run these commands from the project root with the virtual environment activated.

Task 1 – Cosine similarity

py .\task1_embeddings\cosine_similarity.py

Task 2 – Chunking methods

py .\task2_chunking\chunking_methods.py

Task 2 – Stable ID verification

py .\task2_chunking\stable_id_edit_test.py

Task 3 – FAISS benchmark

py .\task3_vector_storage\faiss_benchmark.py

Task 3 – PDF filtering

py .\task3_vector_storage\qdrant_pdf_filter.py

Security and Important Notes

.env is excluded from Git using .gitignore.

The virtual environment is excluded from Git.

Local Qdrant database files are excluded from Git.

Do not commit API keys, passwords, or other secrets.

The API's development authentication key should be replaced with a securely configured environment variable before production deployment.

Local vector database data may need to be recreated after cloning the repository.

Learning Outcomes

Through this project, I learned:

How text is converted into numerical embeddings.

How cosine similarity measures semantic similarity.

How to extract and split documents into chunks.

How to maintain stable IDs when documents change.

How vector databases store and retrieve embeddings.

How exact and approximate vector search differ.

How to build APIs using FastAPI.

How to implement API authentication.

How to test APIs using Pytest.

How to evaluate semantic search performance using Hit@K and MRR.

Author

Srinivasan

GitHub: https://github.com/Srinivasan217

Repository

https://github.com/Srinivasan217/Semantic-Search-documents