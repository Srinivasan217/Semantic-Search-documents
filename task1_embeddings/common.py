from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


SENTENCES = [
     "Artificial intelligence helps computers learn patterns from data.",
    "Machine learning allows computer systems to identify patterns in information.",
    "Neural networks are used to recognize images and speech.",
    "Data scientists train models using historical datasets.",
    "The automobile consumes petrol to move.",
    "A car needs gasoline for transportation.",
    "A train carries passengers between cities.",
    "Employees must reset their password every 90 days.",
    "Staff should change their login password every three months.",
    "Workers can reserve meeting rooms through the company calendar."
]

CATEGORIES = [
    "Technology & AI",
    "Technology & AI",
    "Technology & AI",
    "Technology & AI",
    "Transport",
    "Transport",
    "Transport",
    "Workplace Policies",
    "Workplace Policies",
    "Workplace Policies"
]
def load_model():
    return SentenceTransformer(MODEL_NAME)
