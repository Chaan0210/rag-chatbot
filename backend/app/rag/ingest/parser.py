import pymupdf4llm
from sqlalchemy.orm import Session
from app.db.models.documents import Document, Chunk
from app.core.config import settings
from openai import OpenAI
import json

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def get_embedding(text: str):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=settings.EMBEDDING_MODEL).data[0].embedding

def process_pdf(file_path: str, filename: str, db: Session):
    md_text = pymupdf4llm.to_markdown(file_path, page_chunks=True)
    
    db_doc = Document(filename=filename)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    
    chunks_to_add = []
    
    for page in md_text:
        page_num = page['metadata']['page']
        content = page['text']
        
        meta = {
            "source": filename,
            "page": page_num,
            "year": "2024" if "2024" in filename else "2025",
            "quarter": "1Q" if "1Q" in filename else "2Q"
        }
        
        header = f"File: {filename}, Page: {page_num}\n\n"
        final_content = header + content
        
        embedding_vector = get_embedding(final_content)
        
        chunk = Chunk(
            document_id=db_doc.id,
            content=final_content,
            page_number=page_num,
            embedding=embedding_vector,
            metadata_json=json.dumps(meta)
        )
        chunks_to_add.append(chunk)
        
    db.add_all(chunks_to_add)
    db.commit()