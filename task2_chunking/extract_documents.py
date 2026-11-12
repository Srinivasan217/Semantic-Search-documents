
from pathlib import Path
import json
import pymupdf
from docx import Document

DOCUMENTS_DIR = Path(__file__).parent / "documents"
OUTPUT_DIR = Path(__file__).parent / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

extracted_documents = []

for file_path in DOCUMENTS_DIR.iterdir():

    if not file_path.is_file():
        continue

    suffix = file_path.suffix.lower()

    print(f"\nProcessing: {file_path.name}")

    if suffix == ".txt":
        text = file_path.read_text(encoding="utf-8")

        extracted_documents.append({
            "source": file_path.name,
            "doc_type": "txt",
            "page": 1,
            "text": text
        })

    elif suffix == ".pdf":
        pdf = pymupdf.open(file_path)

        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            text = page.get_text()

            extracted_documents.append({
                "source": file_path.name,
                "doc_type": "pdf",
                "page": page_index + 1,
                "text": text
            })

        pdf.close()

    elif suffix == ".docx":
        document = Document(file_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        extracted_documents.append({
            "source": file_path.name,
            "doc_type": "docx",
            "page": None,
            "text": text
        })

    else:
        print("Unsupported file format. Skipping.")

output_path = OUTPUT_DIR / "extracted_documents.json"

with output_path.open("w", encoding="utf-8") as file:
    json.dump(extracted_documents, file, indent=4, ensure_ascii=False)

print("\nText extraction completed!")
print(f"Documents/pages extracted: {len(extracted_documents)}")
print(f"Saved to: {output_path}")
