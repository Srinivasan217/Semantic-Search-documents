
from pathlib import Path
import fitz
from docx import Document

# Documents folder
DOCUMENTS_DIR = Path(__file__).parent / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)

# 1. Create PDF document
pdf_path = DOCUMENTS_DIR / "technical_document.pdf"

pdf = fitz.open()
page = pdf.new_page()

pdf_text = """
Technical Document

Document Search Architecture

A document search system processes different file formats.
Text extraction converts PDF and Word documents into plain text.

Text Chunking

Large documents are divided into smaller text chunks.
Each chunk contains a limited number of tokens.
Overlapping chunks help preserve context between sections.

Vector Embeddings

Embedding models convert text into numerical vectors.
Similar meanings produce vectors that are close together.

Search Process

The search system converts user queries into embeddings.
It compares query vectors with stored document vectors.
The most relevant chunks are returned to the user.
"""

page.insert_text((72, 72), pdf_text, fontsize=11)
pdf.save(pdf_path)
pdf.close()

# 2. Create Word document
docx_path = DOCUMENTS_DIR / "training_manual.docx"

document = Document()
document.add_heading("Training Manual", level=1)

document.add_heading("Introduction", level=2)
document.add_paragraph(
    "This manual introduces employees to the document search system."
)

document.add_heading("Document Upload", level=2)
document.add_paragraph(
    "Users can upload PDF, Word, and text documents. "
    "The system extracts text from each document."
)

document.add_heading("Chunking Process", level=2)
document.add_paragraph(
    "Documents are divided into smaller chunks. "
    "Each chunk stores metadata such as source, page, and chunk index."
)

document.add_heading("Search Process", level=2)
document.add_paragraph(
    "The system uses embeddings to identify relevant information. "
    "Users receive the most relevant document sections."
)

document.save(docx_path)

print("PDF and DOCX documents created successfully!")
print(f"PDF: {pdf_path}")
print(f"DOCX: {docx_path}")
