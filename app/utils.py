import torch
def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"Cached:   {torch.cuda.memory_reserved()/1024**2:.2f} MB")