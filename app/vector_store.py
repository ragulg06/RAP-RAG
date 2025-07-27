# app/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
import numpy as np
from typing import List, Dict, Optional

class AdvancedVectorStore:
    def __init__(self, collection_name="advanced_rag_docs"):
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self._create_collection()
    
    def _create_collection(self):
        """Create collection with optimized settings"""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            # Optimize for memory usage
            optimizers_config={
                "default_segment_number": 2
            }
        )
    
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray):
        """Add documents with enhanced metadata"""
        points = []
        
        for chunk, embedding in zip(chunks, embeddings):
            # Enhanced payload with more metadata for filtering
            payload = {
                **chunk["metadata"],
                "text": chunk["text"],
                "text_length": len(chunk["text"]),
                "word_count": len(chunk["text"].split())
            }
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Batch upsert for efficiency
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filename_filter: Optional[str] = None) -> List:
        """Enhanced search with filtering and scoring"""
        
        # Prepare filter if filename is specified
        search_filter = None
        if filename_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename_filter)
                    )
                ]
            )
        
        # Perform search with increased top_k for better coverage
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter=search_filter,
            limit=top_k * 2,  # Get more results for re-ranking
            score_threshold=0.3  # Filter out very low similarity scores
        )
        
        # Re-rank results based on multiple factors
        ranked_hits = self._rerank_results(hits, top_k)
        
        return ranked_hits
    
    def _rerank_results(self, hits: List, top_k: int) -> List:
        """Re-rank results based on multiple factors"""
        if not hits:
            return hits
        
        # Score based on multiple factors
        for hit in hits:
            base_score = hit.score
            text_length_bonus = min(hit.payload.get('text_length', 0) / 1000, 0.1)
            word_count_bonus = min(hit.payload.get('word_count', 0) / 100, 0.05)
            
            # Boost score for longer, more informative chunks
            hit.score = base_score + text_length_bonus + word_count_bonus
        
        # Sort by enhanced score and return top_k
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]
