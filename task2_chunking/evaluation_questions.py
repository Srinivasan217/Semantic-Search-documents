
import json
from pathlib import Path

questions = [
    {
        "question": "How often must employees reset their passwords?",
        "expected_page": 1
    },
    {
        "question": "How many paid leave days do employees receive annually?",
        "expected_page": 2
    },
    {
        "question": "What are the normal office working hours?",
        "expected_page": 3
    },
    {
        "question": "Which file formats can users upload?",
        "expected_page": 4
    },
    {
        "question": "Why does the system divide documents into smaller sections?",
        "expected_page": 5
    },
    {
        "question": "How many dimensions does all-MiniLM-L6-v2 generate?",
        "expected_page": 6
    },
    {
        "question": "How does semantic search find relevant information?",
        "expected_page": 7
    },
    {
        "question": "What information is stored in a vector database record?",
        "expected_page": 8
    },
    {
        "question": "Which API endpoint performs document searching?",
        "expected_page": 9
    },
    {
        "question": "Where should API keys and secret credentials be stored?",
        "expected_page": 10
    }
]

output_path = (
    Path(__file__).parent
    / "output"
    / "evaluation_questions.json"
)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(questions, file, indent=4)

print("Evaluation questions created successfully!")
print(f"Total questions: {len(questions)}")

for index, item in enumerate(questions, start=1):
    print(
        f"Q{index}: {item['question']} "
        f"| Expected page: {item['expected_page']}"
    )

print(f"\nSaved to: {output_path}")
