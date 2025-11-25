#!/usr/bin/env python3
"""
Performance monitoring script for cookie tracker
Tracks FPS, memory usage, and processing times to verify optimizations
"""

import psutil
import time
import threading
import sys
from datetime import datetime


class PerformanceMonitor:
    def __init__(self, process_name="run_tracker.py"):
        self.process_name = process_name
        self.monitoring = False
        self.stats = {
            'fps_history': [],
            'memory_history': [],
            'cpu_history': [],
            'start_time': None,
            'frame_count': 0,
            'last_frame_time': None
        }
        self.monitor_thread = None

    def start_monitoring(self):
        """Start monitoring performance metrics"""
        self.monitoring = True
        self.stats['start_time'] = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Performance monitoring started...")

    def stop_monitoring(self):
        """Stop monitoring and return summary"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        return self._generate_summary()

    def update_frame_count(self):
        """Call this method each time a frame is processed"""
        current_time = time.time()
        self.stats['frame_count'] += 1
        
        if self.stats['last_frame_time'] is not None:
            frame_time = current_time - self.stats['last_frame_time']
            if frame_time > 0:
                fps = 1.0 / frame_time
                self.stats['fps_history'].append(fps)
                # Keep only last 100 FPS measurements
                if len(self.stats['fps_history']) > 100:
                    self.stats['fps_history'].pop(0)
        
        self.stats['last_frame_time'] = current_time

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Find tracker process
                tracker_process = None
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info['cmdline']
                        if cmdline and any(self.process_name in cmd for cmd in cmdline):
                            tracker_process = psutil.Process(proc.info['pid'])
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                if tracker_process:
                    # Get memory usage
                    memory_info = tracker_process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024  # RSS in MB
                    
                    # Get CPU usage
                    cpu_percent = tracker_process.cpu_percent(interval=0.1)
                    
                    # Store metrics
                    self.stats['memory_history'].append(memory_mb)
                    self.stats['cpu_history'].append(cpu_percent)
                    
                    # Keep only last 100 measurements
                    if len(self.stats['memory_history']) > 100:
                        self.stats['memory_history'].pop(0)
                    if len(self.stats['cpu_history']) > 100:
                        self.stats['cpu_history'].pop(0)

            except Exception as e:
                pass  # Continue monitoring even if there are errors
            
            time.sleep(1)  # Monitor every second

    def _generate_summary(self):
        """Generate performance summary"""
        if not self.stats['start_time']:
            return "No monitoring data available"

        total_time = time.time() - self.stats['start_time']
        
        summary = []
        summary.append("=" * 60)
        summary.append("🚀 PERFORMANCE SUMMARY")
        summary.append("=" * 60)
        
        # Overall metrics
        summary.append(f"Total Runtime:     {total_time:.1f} seconds")
        summary.append(f"Total Frames:      {self.stats['frame_count']}")
        
        if total_time > 0:
            avg_fps = self.stats['frame_count'] / total_time
            summary.append(f"Average FPS:       {avg_fps:.2f}")
        
        # FPS analysis
        if self.stats['fps_history']:
            fps_history = self.stats['fps_history']
            summary.append(f"Current FPS:       {fps_history[-1]:.2f}")
            summary.append(f"Max FPS:           {max(fps_history):.2f}")
            summary.append(f"Min FPS:           {min(fps_history):.2f}")
            summary.append(f"Avg FPS (recent):  {sum(fps_history) / len(fps_history):.2f}")
        
        # Memory analysis
        if self.stats['memory_history']:
            memory_history = self.stats['memory_history']
            summary.append(f"Current Memory:    {memory_history[-1]:.1f} MB")
            summary.append(f"Max Memory:        {max(memory_history):.1f} MB")
            summary.append(f"Min Memory:        {min(memory_history):.1f} MB")
            summary.append(f"Avg Memory:        {sum(memory_history) / len(memory_history):.1f} MB")
        
        # CPU analysis
        if self.stats['cpu_history']:
            cpu_history = self.stats['cpu_history']
            summary.append(f"Current CPU:       {cpu_history[-1]:.1f}%")
            summary.append(f"Max CPU:           {max(cpu_history):.1f}%")
            summary.append(f"Avg CPU:           {sum(cpu_history) / len(cpu_history):.1f}%")
        
        # Performance assessment
        summary.append("")
        summary.append("📊 PERFORMANCE ASSESSMENT:")
        
        if self.stats['fps_history']:
            recent_fps = sum(self.stats['fps_history'][-10:]) / min(len(self.stats['fps_history']), 10)
            if recent_fps >= 25:
                summary.append("✅ Excellent performance (25+ FPS)")
            elif recent_fps >= 15:
                summary.append("🟡 Good performance (15-25 FPS)")
            elif recent_fps >= 10:
                summary.append("🟠 Acceptable performance (10-15 FPS)")
            else:
                summary.append("🔴 Poor performance (<10 FPS)")
        
        if self.stats['memory_history']:
            recent_memory = self.stats['memory_history'][-1]
            if recent_memory < 500:
                summary.append("✅ Low memory usage (<500 MB)")
            elif recent_memory < 1000:
                summary.append("🟡 Moderate memory usage (500-1000 MB)")
            else:
                summary.append("🟠 High memory usage (>1000 MB)")
        
        summary.append("=" * 60)
        
        return "\n".join(summary)

    def print_realtime_stats(self):
        """Print real-time performance stats"""
        if not self.monitoring:
            return
            
        fps = self.stats['fps_history'][-1] if self.stats['fps_history'] else 0
        memory = self.stats['memory_history'][-1] if self.stats['memory_history'] else 0
        cpu = self.stats['cpu_history'][-1] if self.stats['cpu_history'] else 0
        
        print(f"\r🔄 FPS: {fps:6.2f} | Memory: {memory:6.1f}MB | CPU: {cpu:5.1f}% | Frames: {self.stats['frame_count']}", 
              end='', flush=True)


def main():
    """Standalone performance monitoring"""
    if len(sys.argv) < 2:
        print("Usage: python performance_monitor.py <video_path>")
        print("This will run the tracker with performance monitoring")
        return
    
    video_path = sys.argv[1]
    monitor = PerformanceMonitor()
    
    print("🚀 Starting cookie tracker with performance monitoring...")
    print("Press Ctrl+C to stop and view performance summary")
    
    monitor.start_monitoring()
    
    try:
        # Import and run the tracker
        import run_tracker
        import threading
        import time
        
        # Run tracker in a separate thread
        def run_tracker_thread():
            try:
                run_tracker.main(video_path)
            except Exception as e:
                print(f"Tracker error: {e}")
        
        tracker_thread = threading.Thread(target=run_tracker_thread)
        tracker_thread.daemon = True
        tracker_thread.start()
        
        # Monitor in main thread
        start_time = time.time()
        while tracker_thread.is_alive():
            monitor.print_realtime_stats()
            time.sleep(0.5)
            
            # Auto-update frame count estimation based on time
            if time.time() - start_time > 1:
                monitor.update_frame_count()
        
        print("\n\nTracker finished!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n\nError: {e}")
    
    finally:
        summary = monitor.stop_monitoring()
        print(summary)


if __name__ == "__main__":
    main() 