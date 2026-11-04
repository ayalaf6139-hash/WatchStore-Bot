import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "watchstore_docs"

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "4"))
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.55"))
CONDENSE_QUESTION = os.getenv("CONDENSE_QUESTION", "true").lower() == "true"

NO_ANSWER_TEXT = "לא מצאתי מידע על כך במסמכי WatchStore. אפשר לפנות לשירות הלקוחות בטלפון 03-5550000."
