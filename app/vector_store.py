from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class VectorStore:
    def __init__(self, collection_name="rag_docs"):
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def add_documents(self, chunks, embeddings):
        points = []
        for chunk, vector in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=chunk["metadata"] | {"text": chunk["text"]}
            )
            points.append(point)
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding, top_k=3):
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        return hits