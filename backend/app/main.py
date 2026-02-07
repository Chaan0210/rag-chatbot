from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.db.models import chat, documents
from app.rag.ingest.parser import process_pdf
from app.rag.retrieve.hybrid import hybrid_search
from app.rag.generate.engine import generate_response
from app.db.database import get_db
import shutil
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_location = f"data/pdfs/{file.filename}"
    os.makedirs("data/pdfs", exist_ok=True)
    with open(file_location, "wb+") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process PDF
    process_pdf(file_location, file.filename, db)
    return {"status": "success", "filename": file.filename}

@app.post("/api/chat")
async def chat_endpoint(request: dict, db: Session = Depends(get_db)):
    user_query = request.get("message")
    # 1. Multi-turn context handling (TODO: Summary of history)
    
    # 2. Retrieval
    chunks = hybrid_search(user_query, db)
    
    # 3. Generation
    response_data = await generate_response(user_query, chunks)
    
    # 4. Return (Frontend handles streaming or JSON parsing)
    return response_data