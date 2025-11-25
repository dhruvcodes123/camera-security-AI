#!/usr/bin/env python3
"""
Performance Monitor for Person Re-Identification System
Monitors GPU usage, memory, and processing performance in real-time.
"""

import torch
import psutil
import time
import threading
from typing import Dict, List
import numpy as np

class PerformanceMonitor:
    """Real-time performance monitoring for the Re-ID system."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        """
        Initialize performance monitor.
        
        Args:
            monitoring_interval: Seconds between monitoring updates
        """
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.performance_history = []
        self.max_history_size = 1000
        
        # Performance metrics
        self.gpu_usage_history = []
        self.memory_usage_history = []
        self.cpu_usage_history = []
        self.fps_history = []
        
        # print("✅ Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        if self.is_monitoring:
            # print("⚠️ Performance monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        # print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        # print("📊 Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect performance metrics
                metrics = self._collect_metrics()
                self.performance_history.append(metrics)
                
                # Keep history size manageable
                if len(self.performance_history) > self.max_history_size:
                    self.performance_history.pop(0)
                
                # Log critical issues
                self._check_critical_issues(metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                # print(f"⚠️ Performance monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> Dict:
        """Collect current performance metrics."""
        metrics = {
            'timestamp': time.time(),
            'gpu_memory_usage': 0.0,
            'gpu_memory_allocated': 0.0,
            'gpu_memory_reserved': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'fps': 0.0
        }
        
        # GPU metrics
        if torch.cuda.is_available():
            try:
                metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                metrics['gpu_memory_usage'] = (metrics['gpu_memory_allocated'] / total_gpu_memory) * 100
            except Exception as e:
                # print(f"⚠️ GPU metrics error: {e}")
                pass
        
        # CPU and system memory
        try:
            metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            metrics['memory_usage'] = memory.percent
        except Exception as e:
            print(f"⚠️ System metrics error: {e}")
        
        # Calculate FPS from recent history
        if len(self.fps_history) > 1:
            recent_fps = self.fps_history[-10:]  # Last 10 FPS measurements
            metrics['fps'] = np.mean(recent_fps) if recent_fps else 0.0
        
        return metrics
    
    def _check_critical_issues(self, metrics: Dict):
        """Check for critical performance issues."""
        warnings = []
        
        # GPU memory warnings
        if metrics['gpu_memory_usage'] > 90:
            warnings.append(f"🚨 CRITICAL GPU Memory: {metrics['gpu_memory_usage']:.1f}%")
        elif metrics['gpu_memory_usage'] > 80:
            warnings.append(f"⚠️ HIGH GPU Memory: {metrics['gpu_memory_usage']:.1f}%")
        
        # CPU warnings
        if metrics['cpu_usage'] > 90:
            warnings.append(f"🚨 CRITICAL CPU Usage: {metrics['cpu_usage']:.1f}%")
        elif metrics['cpu_usage'] > 80:
            warnings.append(f"⚠️ HIGH CPU Usage: {metrics['cpu_usage']:.1f}%")
        
        # System memory warnings
        if metrics['memory_usage'] > 90:
            warnings.append(f"🚨 CRITICAL System Memory: {metrics['memory_usage']:.1f}%")
        elif metrics['memory_usage'] > 80:
            warnings.append(f"⚠️ HIGH System Memory: {metrics['memory_usage']:.1f}%")
        
        # FPS warnings
        if metrics['fps'] < 10:
            warnings.append(f"🚨 CRITICAL Low FPS: {metrics['fps']:.1f}")
        elif metrics['fps'] < 20:
            warnings.append(f"⚠️ LOW FPS: {metrics['fps']:.1f}")
        
        # Print warnings
        if warnings:
            # print(f"📊 Performance Warnings at {time.strftime('%H:%M:%S')}:")
            # for warning in warnings:
            #     print(f"   {warning}")
            pass
    
    def update_fps(self, fps: float):
        """Update FPS measurement."""
        self.fps_history.append(fps)
        if len(self.fps_history) > 50:  # Keep last 50 FPS measurements
            self.fps_history.pop(0)
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary."""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-10:]  # Last 10 measurements
        
        summary = {
            'current_gpu_memory_gb': recent_metrics[-1]['gpu_memory_allocated'],
            'current_gpu_usage_percent': recent_metrics[-1]['gpu_memory_usage'],
            'current_cpu_usage_percent': recent_metrics[-1]['cpu_usage'],
            'current_memory_usage_percent': recent_metrics[-1]['memory_usage'],
            'current_fps': recent_metrics[-1]['fps'],
            'avg_gpu_memory_gb': np.mean([m['gpu_memory_allocated'] for m in recent_metrics]),
            'avg_gpu_usage_percent': np.mean([m['gpu_memory_usage'] for m in recent_metrics]),
            'avg_cpu_usage_percent': np.mean([m['cpu_usage'] for m in recent_metrics]),
            'avg_fps': np.mean([m['fps'] for m in recent_metrics]),
            'max_gpu_memory_gb': max([m['gpu_memory_allocated'] for m in recent_metrics]),
            'max_gpu_usage_percent': max([m['gpu_memory_usage'] for m in recent_metrics])
        }
        
        return summary
    
    def print_performance_summary(self):
        """Print current performance summary."""
        summary = self.get_performance_summary()
        if not summary:
            print("📊 No performance data available")
            return
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        print(f"🎯 Current FPS: {summary['current_fps']:.1f}")
        print(f"📈 Average FPS: {summary['avg_fps']:.1f}")
        print(f"🔥 GPU Memory: {summary['current_gpu_memory_gb']:.2f} GB ({summary['current_gpu_usage_percent']:.1f}%)")
        print(f"📊 Average GPU Usage: {summary['avg_gpu_usage_percent']:.1f}%")
        print(f"💾 System Memory: {summary['current_memory_usage_percent']:.1f}%")
        print(f"🖥️  CPU Usage: {summary['current_cpu_usage_percent']:.1f}%")
        print(f"📈 Peak GPU Memory: {summary['max_gpu_memory_gb']:.2f} GB ({summary['max_gpu_usage_percent']:.1f}%)")
        print("="*60)
    
    def get_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        summary = self.get_performance_summary()
        if not summary:
            return ["No performance data available"]
        
        recommendations = []
        
        # GPU memory recommendations
        if summary['current_gpu_usage_percent'] > 90:
            recommendations.append("🚨 CRITICAL: Reduce batch size or number of concurrent persons")
        elif summary['current_gpu_usage_percent'] > 80:
            recommendations.append("⚠️ HIGH GPU Memory: Consider reducing batch size")
        
        # FPS recommendations
        if summary['current_fps'] < 15:
            recommendations.append("🚨 CRITICAL: System is too slow - reduce processing complexity")
        elif summary['current_fps'] < 25:
            recommendations.append("⚠️ LOW FPS: Consider optimizing processing pipeline")
        
        # CPU recommendations
        if summary['current_cpu_usage_percent'] > 90:
            recommendations.append("🚨 CRITICAL: CPU overloaded - reduce processing load")
        elif summary['current_cpu_usage_percent'] > 80:
            recommendations.append("⚠️ HIGH CPU: Consider using more GPU processing")
        
        if not recommendations:
            recommendations.append("✅ Performance is within acceptable limits")
        
        return recommendations

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def start_performance_monitoring():
    """Start global performance monitoring."""
    performance_monitor.start_monitoring()

def stop_performance_monitoring():
    """Stop global performance monitoring."""
    performance_monitor.stop_monitoring()

def update_fps(fps: float):
    """Update FPS measurement."""
    performance_monitor.update_fps(fps)

def print_performance_summary():
    """Print current performance summary."""
    performance_monitor.print_performance_summary()

def get_performance_recommendations():
    """Get performance recommendations."""
    return performance_monitor.get_recommendations() 