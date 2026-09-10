import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from ingestion.extractor import DocumentExtractor
from vector_store.store import VectorStore
from retrieval.retriever import SemanticRetriever

app = FastAPI(title="Knowledge Assistant API", version="1.0.0")
extractor = DocumentExtractor()
vector_store = VectorStore()
try:
    vector_store.load()
except FileNotFoundError:
    pass
retriever = SemanticRetriever(vector_store)

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv'}
UPLOAD_DIR = "data/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "ok"}

def simple_chunk(text: str, chunk_size: int = 500):
    if not text:
        return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {ext}. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    document_id = str(uuid.uuid4())
    temp_file_path = os.path.join(UPLOAD_DIR, f"{document_id}{ext}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = extractor.extract(temp_file_path)
        raw_text = result.get('text', '')
        metadata = result.get('metadata', {})
        
        chunks = simple_chunk(raw_text)
        
        chunk_docs = []
        for i, chunk_text in enumerate(chunks):
            chunk_docs.append({
                "chunk_id": f"{document_id}_{i}",
                "document_id": document_id,
                "chunk_index": i,
                "document_name": file.filename,
                "source_location": temp_file_path,
                "text": chunk_text
            })
            
        vector_store.add_chunks(chunk_docs)
        vector_store.save()
        
        return {
            "document_id": document_id,
            "filename": file.filename,
            "file_type": ext,
            "chunk_count": len(chunks),
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/retrieve")
async def retrieve_knowledge(request: RetrieveRequest):
    try:
        results = retriever.retrieve(request.query, top_k=request.top_k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
