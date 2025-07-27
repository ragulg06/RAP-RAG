# app/embedder.py
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List

class AdvancedEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(f'models/{model_name}', device=self.device)
        
        # Optimize for T4 GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed documents in batches to manage memory"""
        batch_size = 32  # Optimal for T4 GPU
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings.cpu())
            
            # Clear GPU cache between batches
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return torch.cat(embeddings, dim=0)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with optimization"""
        # Query preprocessing for better retrieval
        processed_query = self._preprocess_query(query)
        
        embedding = self.model.encode(
            processed_query, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        return embedding.cpu()
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess query to improve retrieval"""
        # Convert to lower case and clean
        query = query.lower().strip()
        
        # Add question words if not present (helps with retrieval)
        question_starters = ['what', 'how', 'when', 'where', 'why', 'which', 'who']
        if not any(query.startswith(starter) for starter in question_starters):
            # Try to infer question type and add appropriate starter
            if 'policy' in query or 'rule' in query:
                query = f"what is the {query}"
            elif 'time' in query or 'duration' in query:
                query = f"how long {query}"
            elif 'contact' in query or 'reach' in query:
                query = f"how to {query}"
        
        return query
