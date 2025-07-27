# Inside document_loader.py, indentations fixed

import PyPDF2
import docx
import os
import re
from typing import List, Dict

class AdvancedDocumentLoader:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_and_chunk_documents(self, filepath: str) -> List[Dict]:
        """Load and chunk documents with better file format detection"""
        filepath_lower = filepath.lower()
        
        if filepath_lower.endswith('.pdf'):
            return self._chunk_pdf(filepath)
        elif filepath_lower.endswith('.docx') or filepath_lower.endswith('.doc'):
            return self._chunk_docx(filepath)
        else:
            # Try to detect file type by content or allow PDF as default
            try:
                return self._chunk_pdf(filepath)
            except Exception as e:
                raise ValueError(f"Unsupported file format: {filepath}. Supported formats: .pdf, .docx, .doc")

    def _chunk_pdf(self, filepath: str) -> List[Dict]:
        """Chunk PDF with better error handling"""
        try:
            reader = PyPDF2.PdfReader(filepath)
            chunks = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue
                    
                cleaned_text = self._clean_text(text)
                page_chunks = self._semantic_chunking(cleaned_text, page_num + 1, filepath)
                chunks.extend(page_chunks)
            
            if not chunks:
                raise ValueError("No text content extracted from PDF")
                
            return chunks
            
        except Exception as e:
            raise ValueError(f"Error processing PDF file: {str(e)}")

    def _chunk_docx(self, filepath: str) -> List[Dict]:
        """Chunk DOCX with better error handling"""
        try:
            doc = docx.Document(filepath)
            full_text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            text = '\n'.join(full_text)
            if not text.strip():
                raise ValueError("No text content extracted from DOCX")
                
            cleaned_text = self._clean_text(text)
            return self._semantic_chunking(cleaned_text, 1, filepath)
            
        except Exception as e:
            raise ValueError(f"Error processing DOCX file: {str(e)}")

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = ''.join(char for char in text if ord(char) < 127 or char.isspace())
        return text.strip()

    def _semantic_chunking(self, text: str, page_num: int, filepath: str) -> List[Dict]:
        """Chunk text at sentence boundaries while respecting size limits"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": {
                        "filename": os.path.basename(filepath),
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "char_count": len(current_chunk.strip())
                    }
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                current_chunk = overlap_text + " " + sentence
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk if it exists
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "metadata": {
                    "filename": os.path.basename(filepath),
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "char_count": len(current_chunk.strip())
                }
            })
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= overlap_chars:
            return text
        return text[-overlap_chars:]
