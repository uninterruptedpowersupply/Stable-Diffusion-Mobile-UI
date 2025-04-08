# memory_utils.py
import gc
import torch
import os
import platform # Import platform

# Set memory management strategy early, before torch initializes CUDA context
# 'expandable_segments:True' is often recommended for PyTorch >= 1.11
# 'max_split_size_mb' can prevent fragmentation but might slightly increase peak usage.
# Adjust based on testing. 64 or 128 are common values.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64" # Alternative
print(f"PYTORCH_CUDA_ALLOC_CONF set to: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")


def configure_system():
    """ Apply system-level performance configurations. """
    print("Configuring system settings...")
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for potentially faster (but slightly less deterministic) runtime
        torch.backends.cudnn.benchmark = True
        # Allow TF32 on Ampere GPUs and newer for faster matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA backend configured (cuDNN benchmark, TF32 allowed).")
    else:
        print("CUDA not available, running on CPU.")

def memory_cleanup():
    """ Perform garbage collection and clear CUDA cache. """
    # print("Performing memory cleanup...") # Make less verbose
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Optional: Reset peak memory stats if you want to track usage per generation
        # torch.cuda.reset_peak_memory_stats()
        # print("CUDA cache cleared.") # Make less verbose
    # else:
        # print("No CUDA cache to clear.")

def get_memory_usage(detail="Peak"):
    """ Returns current or peak memory usage for CUDA device 0. """
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated() / (1024**3) # GB
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3) # GB
        if detail == "Peak":
             return f"Peak VRAM: {peak_mem:.2f} GB"
        elif detail == "Current":
             return f"Current VRAM: {current_mem:.2f} GB"
        else:
             return f"VRAM Now:{current_mem:.2f} GB Peak:{peak_mem:.2f} GB"
    else:
        # Basic RAM usage (less precise for VRAM comparison)
        # Requires 'psutil' package: pip install psutil
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_gb = mem_info.rss / (1024**3) # Resident Set Size in GB
            return f"RAM (RSS): {rss_gb:.2f} GB"
        except ImportError:
            return "CPU Mode (psutil not installed)"