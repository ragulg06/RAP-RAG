from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Target directory to save locally
TARGET_DIR = "models/stablelm-zephyr-3b"
MODEL_ID = "stabilityai/stablelm-zephyr-3b"

# Download tokenizer and model, then save locally
print("🔽 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(TARGET_DIR)

print("🔽 Downloading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
model.save_pretrained(TARGET_DIR)

print(f"✅ Model and tokenizer saved to: {TARGET_DIR}")
