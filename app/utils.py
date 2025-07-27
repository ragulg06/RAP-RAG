# app/utils.py
import torch
import psutil
import time
from typing import Dict, Any

class GPUMonitor:
    """Monitor GPU and system resources"""
    
    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get current GPU and system statistics"""
        stats = {
            "gpu_available": torch.cuda.is_available(),
            "cpu_usage": psutil.cpu_percent(),
            "ram_usage": psutil.virtual_memory().percent,
            "timestamp": time.time()
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "memory_allocated": torch.cuda.memory_allocated(0) / 1024**2,  # MB
                "memory_cached": torch.cuda.memory_reserved(0) / 1024**2,  # MB
                "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            })
        
        return stats
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def optimize_memory():
        """Optimize memory usage"""
        if torch.cuda.is_available():
            # Clear cache
            torch.cuda.empty_cache()
            
            # Set memory fraction if needed
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory < 17 * 1024**3:  # Less than 17GB (T4 has 16GB)
                torch.cuda.set_per_process_memory_fraction(0.9)
