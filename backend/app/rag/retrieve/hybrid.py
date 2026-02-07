from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.models.documents import Chunk
from app.core.config import settings
from sentence_transformers import CrossEncoder
from openai import OpenAI
import json

client = OpenAI(api_key=settings.OPENAI_API_KEY)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') 

def hybrid_search(query: str, db: Session, top_k=50, final_k=5):
    query_embedding = client.embeddings.create(input=[query], model=settings.EMBEDDING_MODEL).data[0].embedding
    
    vector_results = db.query(Chunk).order_by(
        Chunk.embedding.l2_distance(query_embedding)
    ).limit(top_k).all()
    
    pairs = [[query, chunk.content] for chunk in vector_results]
    scores = reranker.predict(pairs)
    
    scored_chunks = list(zip(vector_results, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    final_chunks = [item[0] for item in scored_chunks[:final_k]]
    
    return final_chunks