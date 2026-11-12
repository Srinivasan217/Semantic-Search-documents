
from pathlib import Path
import pymupdf

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "documents"
OUTPUT_DIR.mkdir(exist_ok=True)

pages = [
    {
        "title": "Password Security Policy",
        "content": """
Employees must reset their passwords every 90 days.
Every password must contain at least 12 characters.
Passwords should include uppercase letters, lowercase
letters, numbers, and special characters.

Employees must never share their login credentials.
Multi-factor authentication is mandatory for all
company accounts. The IT department provides password
reset support and helps employees recover locked accounts.

Suspicious login activity must be reported immediately.
Employees should avoid saving passwords on public computers.
The security team reviews access logs regularly.
"""
    },
    {
        "title": "Employee Leave Policy",
        "content": """
Employees receive 20 days of paid leave annually.
Leave requests must be submitted to the reporting manager.
Managers review requests based on team availability.

Employees should submit planned leave requests at least
seven days before the intended leave date.
Emergency leave must be communicated as soon as possible.

The human resources department maintains leave records.
Unused leave is handled according to company policy.
Employees can check their remaining leave balance
through the internal employee portal.
"""
    },
    {
        "title": "Attendance Management",
        "content": """
Employees must record their attendance every working day.
The normal office working hours are 9 AM to 6 PM.
Employees should inform their manager about late arrivals.

Attendance records are maintained by the HR department.
Missing attendance entries should be corrected through
the attendance management portal.

Managers review attendance reports every month.
Repeated attendance issues may require a discussion
with the employee and the HR team.
"""
    },
    {
        "title": "Document Upload System",
        "content": """
The document search application supports PDF, DOCX,
and TXT file formats. Users upload documents through
the document management interface.

After upload, the system identifies the file format
and extracts its readable text. PDF files are processed
page by page. Word files are processed paragraph by
paragraph. Text files are read directly.

Unsupported file formats are rejected with an error.
The original filename is stored as source metadata.
Uploaded documents are prepared for text chunking.
"""
    },
    {
        "title": "Document Chunking",
        "content": """
Document chunking divides long text into smaller sections.
Each section is called a chunk. Chunking helps the search
system retrieve relevant information efficiently.

Fixed-size chunking uses a defined token limit.
Sentence-based chunking tries to preserve complete sentences.
Recursive character splitting uses separators such as
paragraph breaks, line breaks, and spaces.

Overlapping chunks preserve context between neighboring
sections. A common configuration uses 512 characters
with an overlap of 50 characters.
"""
    },
    {
        "title": "Embedding Generation",
        "content": """
Embedding models convert text into numerical vectors.
These vectors represent the semantic meaning of text.

The all-MiniLM-L6-v2 model generates 384-dimensional
embeddings. Embedding models can represent both documents
and user queries in the same vector space.

Similar meanings generally produce vectors with higher
cosine similarity. Embeddings are stored alongside
document metadata so the original source can be identified.

The system generates embeddings for every document chunk
before inserting them into the vector database.
"""
    },
    {
        "title": "Semantic Search",
        "content": """
Semantic search retrieves information based on meaning
rather than requiring exact keyword matches.

A user submits a natural-language question.
The application converts the question into an embedding.
The query vector is compared with stored document vectors.

The system ranks the closest matches by similarity score.
The highest-ranking chunks are returned to the user.
Results include the source document and page information.

Semantic search can identify related expressions even
when the query and document use different words.
"""
    },
    {
        "title": "Vector Database",
        "content": """
A vector database stores numerical embeddings and metadata.
It supports similarity search across large collections
of document chunks.

Each stored record contains a vector, a unique identifier,
and a metadata payload. Metadata can include the source,
page number, document type, and section heading.

Approximate nearest neighbor indexes improve search speed.
Metadata filters can restrict results to selected documents,
file types, or page ranges.

The database supports inserting, updating, searching,
and deleting document records.
"""
    },
    {
        "title": "Search API",
        "content": """
The document search application exposes a FastAPI service.
The POST /search endpoint accepts a user query and top_k.
The top_k value controls the number of returned results.

The API converts the query into an embedding and searches
the vector database. Each result includes its source,
page number, section, similarity score, and text.

The GET /health endpoint reports application status.
The upload endpoint accepts supported document files.
API authentication protects access to search operations.
"""
    },
    {
        "title": "Application Security",
        "content": """
API keys are used to authenticate requests to protected
application endpoints. Secret keys must be stored in
environment variables rather than committed to Git.

The .env file contains local development credentials.
The .gitignore file prevents secret files from being
included in version control.

Input validation checks uploaded files and search queries.
The application should reject unsupported file types
and invalid parameter values.

Error responses should provide useful messages without
revealing passwords, API keys, or internal secrets.
"""
    }
]
# Expand each topic so we can compare all four chunk sizes.
for item in pages:
    original_content = item["content"].strip()

    item["content"] = "\n\n".join(
        f"Section {number}: {item['title']}\n{original_content}"
        for number in range(1, 15)
    )

pdf_path = OUTPUT_DIR / "experiment_corpus.pdf"

pdf = pymupdf.open()

for page_number, item in enumerate(pages, start=1):
    page = pdf.new_page(width=1200, height=2000)
    page_text = (
        f"Page {page_number}: {item['title']}\n\n"
        + item["content"].strip()
    )

    remaining_space = page.insert_textbox(
    pymupdf.Rect(30, 30, 1170, 1970),
    page_text,
    fontsize=5
    )

if remaining_space < 0:
    raise ValueError(f"Text does not fit on page {page_number}."
                     "Reduce the repeated content.")

pdf.save(pdf_path)
pdf.close()

print("Experiment PDF created successfully!")
print(f"Pages created: {len(pages)}")
print(f"Saved to: {pdf_path}")

for page_number, item in enumerate(pages, start=1):
    print(f"Page {page_number}: {item['title']}")
