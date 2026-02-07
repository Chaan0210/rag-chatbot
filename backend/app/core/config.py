import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "RAG Chatbot"
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag_user:rag_password@localhost:5432/rag_db")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL = "text-embedding-3-large"
    LLM_MODEL = "gpt-5"

settings = Settings()