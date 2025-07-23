from sentence_transformers import SentenceTransformer
import os

# Set the directory where you want to save the model
TARGET_DIR = "models/all-MiniLM-L6-v2"

# Download and save the model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model.save(TARGET_DIR)

print(f"✅ Model saved to: {TARGET_DIR}")
