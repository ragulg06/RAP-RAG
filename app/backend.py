# app/backend.py
from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import time
from typing import Optional

from app.document_loader import AdvancedDocumentLoader
from app.embedder import AdvancedEmbedder
from app.vector_store import AdvancedVectorStore
from app.llm_model import LLM
from app.utils import GPUMonitor

# Request models
class QueryRequest(BaseModel):
    query: str
    filename_filter: Optional[str] = None

# Initialize components
app = FastAPI(title="Advanced RAG Chatbot", version="2.0")
loader = AdvancedDocumentLoader()
embedder = AdvancedEmbedder()
vector_store = AdvancedVectorStore()
llm = LLM()
gpu_monitor = GPUMonitor()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_document(file: UploadFile):
    """Upload and process document"""
    try:
        # Save uploaded file
        os.makedirs("data", exist_ok=True)
        
        # Always use a descriptive filename to avoid 'file' issues
        file_extension = os.path.splitext(file.filename)[1] if '.' in file.filename else '.pdf'
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        descriptive_name = f"document_{timestamp}{file_extension}"
        file_path = f"data/{descriptive_name}"
        
        # Debug: Print the actual filename
        print(f"Uploading file: {file.filename}")
        print(f"Using descriptive name: {descriptive_name}")
        print(f"File path: {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        chunks = loader.load_and_chunk_documents(file_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from document")
        
        # Debug: Print the first chunk's filename
        if chunks:
            print(f"First chunk filename: {chunks[0]['metadata']['filename']}")
        
        # Create embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed_documents(texts)
        
        # Add to vector store
        vector_store.add_documents(chunks, embeddings)
        
        # Monitor GPU usage
        gpu_stats = gpu_monitor.get_stats()
        
        return {
            "message": f"Document processed successfully. {len(chunks)} chunks indexed.",
            "chunks_count": len(chunks),
            "gpu_usage": gpu_stats
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/ask/")
async def ask_question(request: QueryRequest):
    """Answer question based on uploaded documents"""
    try:
        start_time = time.time()
        
        # Embed query
        query_embedding = embedder.embed_query(request.query)
        
        # Search for relevant chunks
        hits = vector_store.search(
            query_embedding, 
            top_k=1,  # Changed from 5 to 1 to get only the most relevant source
            filename_filter=request.filename_filter
        )
        
        if not hits:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": "",
                "confidence": 0.0,
                "response_time": time.time() - start_time
            }
        
        # Prepare contexts for LLM (simplified format)
        contexts = []
        sources = []
        for hit in hits:
            base_filename = os.path.splitext(hit.payload['filename'])[0]
            contexts.append({
                "text": hit.payload["text"],
                "source_id": f"{base_filename} | page {hit.payload['page']} | chunk #{hit.payload['chunk_id']}"
            })
            sources.append(f"{base_filename} | page {hit.payload['page']} | chunk #{hit.payload['chunk_id']}")
        
        # Deduplicate sources while preserving order
        unique_sources = list(dict.fromkeys(sources))
        sources_str = "\n".join([f"• {src}" for src in unique_sources])
        
        # Generate answer using current LLM interface
        answer = llm.generate_answer(request.query, contexts)
        
        # Format sources as bulleted list
        # sources_str = "\n".join([f"• {source}" for source in sources]) # This line is now redundant as sources_str is calculated above
        
        # Calculate simple confidence based on answer length and content
        confidence = 0.5  # Default confidence
        if "Answer not found in the document" not in answer:
            confidence = min(0.9, 0.5 + len(answer.split()) / 100)
        
        response_time = time.time() - start_time
        gpu_stats = gpu_monitor.get_stats()
        
        return {
            "answer": answer,
            "sources": sources_str,
            "confidence": confidence,
            "response_time": response_time,
            "gpu_usage": gpu_stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    gpu_stats = gpu_monitor.get_stats()
    return {
        "status": "healthy",
        "gpu_available": gpu_stats["gpu_available"],
        "memory_usage": gpu_stats
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
