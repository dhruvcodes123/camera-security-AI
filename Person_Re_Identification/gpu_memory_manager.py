#!/usr/bin/env python3
"""
GPU Memory Manager for Person Re-Identification System
Optimizes GPU memory usage to prevent OOM errors with multiple persons.
"""

import torch
import gc
import psutil
import time
from typing import Optional

class GPUMemoryManager:
    """Manages GPU memory to prevent OOM errors during multi-person processing."""
    
    def __init__(self, memory_threshold: float = 0.8, cleanup_interval: int = 10):
        """
        Initialize GPU memory manager.
        
        Args:
            memory_threshold: GPU memory usage threshold (0.0-1.0)
            cleanup_interval: Frames between cleanup operations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        self.frame_count = 0
        self.last_cleanup = 0
        
        if torch.cuda.is_available():
            self.total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            # print(f"✅ GPU Memory Manager initialized")
            # print(f"   - Total GPU Memory: {self.total_gpu_memory / 1024**3:.1f} GB")
            # print(f"   - Memory Threshold: {self.memory_threshold * 100:.0f}%")
            # print(f"   - Cleanup Interval: {self.cleanup_interval} frames")
        else:
            # print("⚠️ GPU Memory Manager: CUDA not available, using CPU")
            pass
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage as a percentage."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            allocated = torch.cuda.memory_allocated()
            return allocated / self.total_gpu_memory
        except Exception as e:
            # print(f"⚠️ Error getting GPU memory usage: {e}")
            return 0.0
    
    def get_gpu_memory_info(self) -> dict:
        """Get detailed GPU memory information."""
        if not torch.cuda.is_available():
            return {"usage_percent": 0.0, "allocated_gb": 0.0, "total_gb": 0.0}
        
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = self.total_gpu_memory
            
            return {
                "usage_percent": (allocated / total) * 100,
                "allocated_gb": allocated / 1024**3,
                "reserved_gb": reserved / 1024**3,
                "total_gb": total / 1024**3,
                "free_gb": (total - reserved) / 1024**3
            }
        except Exception as e:
            # print(f"⚠️ Error getting GPU memory info: {e}")
            return {"usage_percent": 0.0, "allocated_gb": 0.0, "total_gb": 0.0}
    
    def should_cleanup(self) -> bool:
        """Check if GPU memory cleanup is needed."""
        if not torch.cuda.is_available():
            return False
        
        self.frame_count += 1
        
        # Check memory threshold
        memory_usage = self.get_gpu_memory_usage()
        threshold_exceeded = memory_usage > self.memory_threshold
        
        # Check cleanup interval
        interval_reached = (self.frame_count - self.last_cleanup) >= self.cleanup_interval
        
        return threshold_exceeded or interval_reached
    
    def cleanup_gpu_memory(self, force: bool = False) -> dict:
        """
        Clean up GPU memory to prevent OOM errors.
        
        Args:
            force: Force cleanup regardless of threshold
            
        Returns:
            dict: Memory usage before and after cleanup
        """
        if not torch.cuda.is_available():
            return {"before": {}, "after": {}, "freed_gb": 0.0}
        
        # Get memory info before cleanup
        before_info = self.get_gpu_memory_info()
        
        if force or self.should_cleanup():
            print(f"🧹 GPU Memory Cleanup - Before: {before_info['allocated_gb']:.2f} GB")
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Small delay to allow memory to be freed
            time.sleep(0.01)
            
            # Get memory info after cleanup
            after_info = self.get_gpu_memory_info()
            
            freed_gb = before_info['allocated_gb'] - after_info['allocated_gb']
            
            print(f"   After: {after_info['allocated_gb']:.2f} GB (Freed: {freed_gb:.2f} GB)")
            
            self.last_cleanup = self.frame_count
            
            return {
                "before": before_info,
                "after": after_info,
                "freed_gb": freed_gb
            }
        
        return {"before": before_info, "after": before_info, "freed_gb": 0.0}
    
    def optimize_for_batch_processing(self, batch_size: int) -> bool:
        """
        Optimize GPU memory for batch processing.
        
        Args:
            batch_size: Number of persons to process simultaneously
            
        Returns:
            bool: True if optimization was successful
        """
        if not torch.cuda.is_available():
            return True
        
        # Estimate memory needed for batch processing
        estimated_memory_per_person = 0.5  # GB per person (rough estimate)
        total_estimated = batch_size * estimated_memory_per_person
        
        current_info = self.get_gpu_memory_info()
        available_memory = current_info['free_gb']
        
        if available_memory < total_estimated:
            print(f"⚠️ Insufficient GPU memory for batch size {batch_size}")
            print(f"   Available: {available_memory:.2f} GB, Needed: {total_estimated:.2f} GB")
            
            # Force cleanup to free more memory
            self.cleanup_gpu_memory(force=True)
            
            # Recheck after cleanup
            current_info = self.get_gpu_memory_info()
            available_memory = current_info['free_gb']
            
            if available_memory < total_estimated:
                print(f"❌ Still insufficient memory after cleanup")
                return False
        
        print(f"✅ GPU memory optimized for batch size {batch_size}")
        return True
    
    def monitor_memory_usage(self) -> None:
        """Monitor and log GPU memory usage."""
        if not torch.cuda.is_available():
            return
        
        info = self.get_gpu_memory_info()
        
        if info['usage_percent'] > 90:
            print(f"🚨 HIGH GPU MEMORY USAGE: {info['usage_percent']:.1f}%")
            print(f"   Allocated: {info['allocated_gb']:.2f} GB")
            print(f"   Free: {info['free_gb']:.2f} GB")
            
            # Force cleanup
            self.cleanup_gpu_memory(force=True)
        elif info['usage_percent'] > 70:
            print(f"⚠️ Moderate GPU memory usage: {info['usage_percent']:.1f}%")
    
    def get_optimal_batch_size(self, max_batch_size: int = 8) -> int:
        """
        Calculate optimal batch size based on available GPU memory.
        
        Args:
            max_batch_size: Maximum allowed batch size
            
        Returns:
            int: Optimal batch size
        """
        if not torch.cuda.is_available():
            return 1
        
        info = self.get_gpu_memory_info()
        available_gb = info['free_gb']
        
        # Estimate memory per person (conservative estimate)
        memory_per_person = 0.8  # GB per person
        
        optimal_batch = int(available_gb / memory_per_person)
        optimal_batch = max(1, min(optimal_batch, max_batch_size))
        
        print(f"📊 Optimal batch size: {optimal_batch} (Available: {available_gb:.2f} GB)")
        
        return optimal_batch 