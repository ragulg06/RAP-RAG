from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.document_loader import load_and_chunk_pdf
from app.embedder import Embedder
from app.vector_store import VectorStore
from app.llm_model import LLM
import shutil, os

app = FastAPI()
embedder = Embedder()
store     = VectorStore()
llm       = LLM()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload/")
async def upload_document(file: UploadFile):
    path = f"data/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    chunks = load_and_chunk_pdf(path)
    embeddings = embedder.embed([c["text"] for c in chunks])
    store.add_documents(chunks, embeddings)
    return {"message": "Document uploaded and indexed."}

@app.post("/ask/")
async def ask_question(query: str):
    # 1) Embed the query and retrieve top‑3 similar chunks
    q_emb = embedder.embed([query])[0]
    hits = store.search(q_emb, top_k=3)

    if not hits:
        return {"answer": "Answer not found in the document.", "source": ""}

    # 2) Concatenate all top-k chunks as context
    contexts = []
    sources = []
    for hit in hits:
        meta = hit.payload
        context = " ".join(meta["text"].split())
        contexts.append(context)
        src = f"{meta['filename']} | page {meta['page']} | chunk #{meta['chunk_id']}"
        sources.append(src)
    full_context = "\n---\n".join(contexts)
    raw_ans = llm.generate_answer(query, full_context).strip()

    # Deduplicate sources and format as a bulleted list
    unique_sources = list(dict.fromkeys(sources))
    sources_str = "\n".join([f"• {src}" for src in unique_sources])

    # 3) If the answer is not found, return fallback with all sources
    if not raw_ans or raw_ans.lower().startswith("answer not found"):
        return {"answer": "Answer not found in the document.", "source": sources_str}

    return {"answer": raw_ans, "source": sources_str}
