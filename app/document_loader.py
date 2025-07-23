from PyPDF2 import PdfReader
import os

def load_and_chunk_pdf(filepath, chunk_size=300, overlap=100):
    reader = PdfReader(filepath)
    chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        start = 0
        chunk_id = 0
        while start < len(text):
            chunk = text[start:start + chunk_size]
            chunks.append({
                "text": chunk,
                "metadata": {
                    "filename": os.path.basename(filepath),
                    "page": page_num + 1,
                    "chunk_id": chunk_id
                }
            })
            start += chunk_size - overlap
            chunk_id += 1
    return chunks