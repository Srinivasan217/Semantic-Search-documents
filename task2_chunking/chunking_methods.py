from pathlib import Path
from datetime import date
import json
import re
import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "output"/"extracted_documents.json"
OUTPUT_DIR = BASE_DIR/"output"

#Load extracted documents
with INPUT_FILE.open("r", encoding="utf-8") as file:
    documents = json.load(file)


#Tokenizer
# 
tokenizer = tiktoken.get_encoding("cl100k_base")

# Method 1: Fixed-size token chunking
def fixed_token_chunking(text, chunk_size=512, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        chunks.append(tokenizer.decode(chunk_tokens))

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks

#METHOD 2: Sentence-based chunking
# 
def sentence_based_chunking(text, max_tokens=512):
    text = re.sub(r"\s+", " ", text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_sentences = []
    current_token_count = 0

    for sentence in sentences:
        if not sentence.strip():
            continue
        sentence_tokens = len(tokenizer.encode(sentence)) 

        if current_sentences and current_token_count + sentence_tokens > max_tokens:   

            chunks.append(" ".join(current_sentences))            
            current_sentences = []
            current_token_count =0

        current_token_count += sentence_tokens
        current_sentences.append(sentence)
    if current_sentences:
          chunks.append(" ".join(current_sentences))
    return chunks


#Method 3: Recursive Character chunking

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n","\n", ".","",""]
)
# Get a section heading from the document text
def get_section_heading(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return "Unknown section"

    heading = lines[0]

    # Remove a PDF page label such as "Page 1:"
    heading = re.sub(r"^Page\s+\d+:\s*", "", heading)

    return heading[:150]


# Create a citation description for a chunk
def describe_chunk(chunk):
    page = chunk["page"]

    if page is None:
        page_text = "Page not specified"
    else:
        page_text = f"Page {page}"

    return (
        f"Source: {chunk['source']} | "
        f"{page_text} | "
        f"Section: {chunk['section_heading']} | "
        f"Chunk: {chunk['chunk_index']}"
    )
results = {
    "fixed_token": [],
    "sentence_based": [],
    "recursive": []
}
# Add metadata to every chunk
created_date = date.today().isoformat()


for document in documents:
    text = document["text"]

    fixed_chunks = fixed_token_chunking(text) 
    sentence_chunks = sentence_based_chunking(text)
    recursive_chunks=  recursive_splitter.split_text(text)

    section_heading = get_section_heading(text)

    methods = {
        "fixed_token":fixed_chunks,
        "sentence_based": sentence_chunks,
        "recursive": recursive_chunks
    }

    for method_name, method_chunks in methods.items():
        for index, chunk_text in enumerate(method_chunks, start=1):

            chunk_data = {
                "source": document["source"],
                "page": document.get("page"),
                "section_heading": section_heading,
                "created_date": created_date,
                "chunk_index": index,
                "doc_type": document["doc_type"],
                "text": chunk_text
            }

            chunk_data["citation"] = describe_chunk(chunk_data)

            results[method_name].append(chunk_data)


print("\nCHUNKING METHOD COMPARISON")
print("-" * 40)

for method, chunks in results.items():
    print(f"{method}: {len(chunks)} chunks")

# Show a sample from each method
for method, chunks in results.items():
     print(f"{method}: {len(chunks)} chunks")


# Show one sample from each method
for method, chunks in results.items():
    print(f"\nSample {method} chunk:")

    if chunks:
        print("Text:", chunks[0]["text"][:200])
        print("Metadata:", {
            "source": chunks[0]["source"],
            "page": chunks[0]["page"],
            "section_heading": chunks[0]["section_heading"],
            "created_date": chunks[0]["created_date"],
            "chunk_index": chunks[0]["chunk_index"],
            "doc_type": chunks[0]["doc_type"]
        })
        print("Citation:", chunks[0]["citation"])


# Save results
output_file = OUTPUT_DIR / "chunking_results.json"

with output_file.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print(f"\nChunking results saved to: {output_file}")
