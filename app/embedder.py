from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('models/all-MiniLM-L6-v2', device=self.device)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)