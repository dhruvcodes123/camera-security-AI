#!/usr/bin/env python3
"""
Gait Cycle Analyzer for Person Re-Identification
Comprehensive gait cycle detection, analysis, and visualization.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional, Union
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import json
import os
from pathlib import Path

@dataclass
class GaitCycle:
    """Data class representing a single gait cycle."""
    start_frame: int
    end_frame: int
    duration: int
    peak_frame: int
    trough_frames: List[int]
    similarity_curve: np.ndarray
    silhouettes: List[np.ndarray]
    confidence: float
    cycle_type: str  # 'normal', 'fast', 'slow', 'irregular'
    features: Dict[str, float]

@dataclass
class GaitAnalysis:
    """Data class containing complete gait analysis results."""
    total_frames: int
    detected_cycles: List[GaitCycle]
    average_cycle_duration: float
    cycle_consistency: float
    gait_speed: str  # 'slow', 'normal', 'fast'
    gait_regularity: str  # 'regular', 'slightly_irregular', 'irregular'
    similarity_curve: np.ndarray
    cycle_boundaries: List[Tuple[int, int]]
    analysis_confidence: float
    recommendations: List[str]

class GaitCycleAnalyzer:
    """
    Comprehensive gait cycle analyzer for person re-identification.
    Detects, analyzes, and visualizes gait cycles from silhouette sequences.
    """
    
    def __init__(self, 
                 min_cycle_frames: int = 15,
                 max_cycle_frames: int = 50,
                 similarity_threshold: float = 0.3,
                 peak_distance: int = 10,
                 smoothing_window: int = 5):
        """
        Initialize the gait cycle analyzer.
        
        Args:
            min_cycle_frames: Minimum frames for a valid gait cycle
            max_cycle_frames: Maximum frames for a valid gait cycle
            similarity_threshold: Threshold for peak detection
            peak_distance: Minimum distance between peaks
            smoothing_window: Window size for signal smoothing
        """
        self.min_cycle_frames = min_cycle_frames
        self.max_cycle_frames = max_cycle_frames
        self.similarity_threshold = similarity_threshold
        self.peak_distance = peak_distance
        self.smoothing_window = smoothing_window
        
        # Analysis parameters
        self.cycle_detection_methods = ['similarity_peaks', 'temporal_analysis', 'morphological']
        self.quality_metrics = ['consistency', 'regularity', 'completeness']
        
        print("✅ Gait Cycle Analyzer initialized")
    
    def analyze_gait_cycles(self, silhouettes: List[np.ndarray], 
                          frame_rate: float = 30.0) -> GaitAnalysis:
        """
        Perform comprehensive gait cycle analysis.
        
        Args:
            silhouettes: List of silhouette images
            frame_rate: Frame rate of the video (fps)
            
        Returns:
            GaitAnalysis: Complete gait analysis results
        """
        if len(silhouettes) < self.min_cycle_frames:
            return self._create_fallback_analysis(silhouettes, "Insufficient frames")
        
        print(f"🔍 Analyzing {len(silhouettes)} frames for gait cycles...")
        
        # Step 1: Compute similarity curve
        similarity_curve = self._compute_similarity_curve(silhouettes)
        
        # Step 2: Detect gait cycles
        detected_cycles = self._detect_gait_cycles(similarity_curve, silhouettes)
        
        # Step 3: Analyze cycle characteristics
        if detected_cycles:
            detected_cycles = self._analyze_cycle_characteristics(detected_cycles, frame_rate)
        
        # Step 4: Compute overall gait metrics
        analysis = self._compute_gait_metrics(detected_cycles, similarity_curve, frame_rate)
        
        print(f"✅ Gait analysis completed: {len(detected_cycles)} cycles detected")
        return analysis
    
    def _compute_similarity_curve(self, silhouettes: List[np.ndarray]) -> np.ndarray:
        """
        Compute similarity curve between consecutive silhouettes.
        
        Args:
            silhouettes: List of silhouette images
            
        Returns:
            np.ndarray: Similarity curve
        """
        similarities = []
        
        # Use first silhouette as reference
        reference = silhouettes[0].flatten().astype(np.float32)
        
        for silhouette in silhouettes:
            # Flatten and normalize silhouette
            current = silhouette.flatten().astype(np.float32)
            
            # Compute cosine similarity
            similarity = 1 - cosine(reference, current)
            similarities.append(similarity)
        
        # Smooth the similarity curve
        similarities = np.array(similarities)
        if len(similarities) > self.smoothing_window:
            similarities = savgol_filter(similarities, self.smoothing_window, 2)
        
        return similarities
    
    def _detect_gait_cycles(self, similarity_curve: np.ndarray, 
                           silhouettes: List[np.ndarray]) -> List[GaitCycle]:
        """
        Detect gait cycles using multiple methods.
        
        Args:
            similarity_curve: Similarity curve
            silhouettes: List of silhouette images
            
        Returns:
            List[GaitCycle]: Detected gait cycles
        """
        cycles = []
        
        # Method 1: Peak-based detection
        peak_cycles = self._detect_cycles_by_peaks(similarity_curve, silhouettes)
        cycles.extend(peak_cycles)
        
        # Method 2: Temporal analysis
        temporal_cycles = self._detect_cycles_by_temporal_analysis(similarity_curve, silhouettes)
        cycles.extend(temporal_cycles)
        
        # Method 3: Morphological analysis
        morphological_cycles = self._detect_cycles_by_morphology(similarity_curve, silhouettes)
        cycles.extend(morphological_cycles)
        
        # Merge and filter overlapping cycles
        merged_cycles = self._merge_overlapping_cycles(cycles)
        
        # Sort by confidence
        merged_cycles.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return top cycles (limit to 5 for performance)
        return merged_cycles[:5]
    
    def _detect_cycles_by_peaks(self, similarity_curve: np.ndarray, 
                               silhouettes: List[np.ndarray]) -> List[GaitCycle]:
        """Detect gait cycles using peak analysis."""
        cycles = []
        
        # Find peaks (high similarity - legs together)
        peaks, peak_properties = find_peaks(
            similarity_curve, 
            distance=self.peak_distance,
            height=self.similarity_threshold,
            prominence=0.1
        )
        
        # Find troughs (low similarity - legs apart)
        inverted_curve = -similarity_curve
        troughs, trough_properties = find_peaks(
            inverted_curve,
            distance=self.peak_distance,
            prominence=0.1
        )
        
        # Combine peaks and troughs to form cycles
        all_points = np.concatenate([peaks, troughs])
        all_points.sort()
        
        for i in range(len(all_points) - 1):
            start_idx = all_points[i]
            end_idx = all_points[i + 1]
            
            # Check if this forms a valid cycle
            if self._is_valid_cycle(start_idx, end_idx, similarity_curve):
                cycle_silhouettes = silhouettes[start_idx:end_idx + 1]
                cycle_curve = similarity_curve[start_idx:end_idx + 1]
                
                # Find peak within this cycle
                cycle_peak_idx = start_idx + np.argmax(cycle_curve)
                
                # Find troughs within this cycle
                cycle_troughs = [t for t in troughs if start_idx <= t <= end_idx]
                
                cycle = GaitCycle(
                    start_frame=start_idx,
                    end_frame=end_idx,
                    duration=end_idx - start_idx + 1,
                    peak_frame=cycle_peak_idx,
                    trough_frames=cycle_troughs,
                    similarity_curve=cycle_curve,
                    silhouettes=cycle_silhouettes,
                    confidence=self._compute_cycle_confidence(cycle_curve),
                    cycle_type='normal',
                    features={}
                )
                
                cycles.append(cycle)
        
        return cycles
    
    def _detect_cycles_by_temporal_analysis(self, similarity_curve: np.ndarray,
                                          silhouettes: List[np.ndarray]) -> List[GaitCycle]:
        """Detect gait cycles using temporal analysis."""
        cycles = []
        
        # Use autocorrelation to find periodic patterns
        autocorr = np.correlate(similarity_curve, similarity_curve, mode='full')
        autocorr = autocorr[len(similarity_curve)-1:]
        
        # Find peaks in autocorrelation (periodic patterns)
        peaks, _ = find_peaks(autocorr[10:], distance=self.peak_distance)
        peaks += 10  # Adjust for the offset
        
        if len(peaks) > 0:
            # Use the first significant peak as cycle length
            cycle_length = peaks[0]
            
            # Extract cycles based on this length
            for start_idx in range(0, len(similarity_curve) - cycle_length, cycle_length):
                end_idx = start_idx + cycle_length
                
                if self._is_valid_cycle(start_idx, end_idx, similarity_curve):
                    cycle_silhouettes = silhouettes[start_idx:end_idx]
                    cycle_curve = similarity_curve[start_idx:end_idx]
                    
                    cycle = GaitCycle(
                        start_frame=start_idx,
                        end_frame=end_idx,
                        duration=cycle_length,
                        peak_frame=start_idx + np.argmax(cycle_curve),
                        trough_frames=[start_idx + np.argmin(cycle_curve)],
                        similarity_curve=cycle_curve,
                        silhouettes=cycle_silhouettes,
                        confidence=0.7,  # Moderate confidence for temporal method
                        cycle_type='temporal',
                        features={'autocorr_peak': autocorr[cycle_length]}
                    )
                    
                    cycles.append(cycle)
        
        return cycles
    
    def _detect_cycles_by_morphology(self, similarity_curve: np.ndarray,
                                   silhouettes: List[np.ndarray]) -> List[GaitCycle]:
        """Detect gait cycles using morphological analysis."""
        cycles = []
        
        # Use morphological operations to find cycles
        # This method looks for specific patterns in the similarity curve
        
        # Find regions with significant variation
        diff_curve = np.diff(similarity_curve)
        significant_changes = np.where(np.abs(diff_curve) > np.std(diff_curve))[0]
        
        if len(significant_changes) >= 2:
            # Group consecutive significant changes
            change_groups = []
            current_group = [significant_changes[0]]
            
            for i in range(1, len(significant_changes)):
                if significant_changes[i] - significant_changes[i-1] <= 5:
                    current_group.append(significant_changes[i])
                else:
                    if len(current_group) > 1:
                        change_groups.append(current_group)
                    current_group = [significant_changes[i]]
            
            if len(current_group) > 1:
                change_groups.append(current_group)
            
            # Create cycles from change groups
            for i in range(len(change_groups) - 1):
                start_idx = change_groups[i][0]
                end_idx = change_groups[i + 1][-1]
                
                if self._is_valid_cycle(start_idx, end_idx, similarity_curve):
                    cycle_silhouettes = silhouettes[start_idx:end_idx + 1]
                    cycle_curve = similarity_curve[start_idx:end_idx + 1]
                    
                    cycle = GaitCycle(
                        start_frame=start_idx,
                        end_frame=end_idx,
                        duration=end_idx - start_idx + 1,
                        peak_frame=start_idx + np.argmax(cycle_curve),
                        trough_frames=[start_idx + np.argmin(cycle_curve)],
                        similarity_curve=cycle_curve,
                        silhouettes=cycle_silhouettes,
                        confidence=0.6,  # Lower confidence for morphological method
                        cycle_type='morphological',
                        features={'change_points': len(change_groups[i])}
                    )
                    
                    cycles.append(cycle)
        
        return cycles
    
    def _is_valid_cycle(self, start_idx: int, end_idx: int, 
                       similarity_curve: np.ndarray) -> bool:
        """Check if a cycle is valid."""
        duration = end_idx - start_idx + 1
        
        # Check duration constraints
        if duration < self.min_cycle_frames or duration > self.max_cycle_frames:
            return False
        
        # Check for sufficient variation
        cycle_curve = similarity_curve[start_idx:end_idx + 1]
        variation = np.std(cycle_curve)
        
        if variation < 0.05:  # Too little variation
            return False
        
        # Check for reasonable similarity range
        if np.max(cycle_curve) - np.min(cycle_curve) < 0.1:
            return False
        
        return True
    
    def _compute_cycle_confidence(self, cycle_curve: np.ndarray) -> float:
        """Compute confidence score for a gait cycle."""
        # Factors that increase confidence:
        # 1. Good variation in similarity curve
        # 2. Smooth curve (low noise)
        # 3. Reasonable duration
        
        variation = np.std(cycle_curve)
        smoothness = 1.0 / (1.0 + np.std(np.diff(cycle_curve)))
        duration_score = min(1.0, len(cycle_curve) / 30.0)  # Prefer ~30 frames
        
        confidence = (variation * 0.4 + smoothness * 0.4 + duration_score * 0.2)
        return min(1.0, confidence)
    
    def _merge_overlapping_cycles(self, cycles: List[GaitCycle]) -> List[GaitCycle]:
        """Merge overlapping or very similar cycles."""
        if not cycles:
            return []
        
        merged = []
        used = set()
        
        for i, cycle1 in enumerate(cycles):
            if i in used:
                continue
            
            merged_cycle = cycle1
            used.add(i)
            
            for j, cycle2 in enumerate(cycles[i+1:], i+1):
                if j in used:
                    continue
                
                # Check for overlap
                overlap_start = max(cycle1.start_frame, cycle2.start_frame)
                overlap_end = min(cycle1.end_frame, cycle2.end_frame)
                
                if overlap_start <= overlap_end:
                    # Overlapping cycles - merge them
                    merged_cycle = GaitCycle(
                        start_frame=min(cycle1.start_frame, cycle2.start_frame),
                        end_frame=max(cycle1.end_frame, cycle2.end_frame),
                        duration=merged_cycle.end_frame - merged_cycle.start_frame + 1,
                        peak_frame=cycle1.peak_frame if cycle1.confidence > cycle2.confidence else cycle2.peak_frame,
                        trough_frames=list(set(cycle1.trough_frames + cycle2.trough_frames)),
                        similarity_curve=np.concatenate([cycle1.similarity_curve, cycle2.similarity_curve]),
                        silhouettes=cycle1.silhouettes + cycle2.silhouettes,
                        confidence=max(cycle1.confidence, cycle2.confidence),
                        cycle_type='merged',
                        features={**cycle1.features, **cycle2.features}
                    )
                    used.add(j)
            
            merged.append(merged_cycle)
        
        return merged
    
    def _analyze_cycle_characteristics(self, cycles: List[GaitCycle], 
                                     frame_rate: float) -> List[GaitCycle]:
        """Analyze characteristics of detected cycles."""
        for cycle in cycles:
            # Analyze cycle duration
            cycle_duration_sec = cycle.duration / frame_rate
            
            if cycle_duration_sec < 0.8:
                cycle.cycle_type = 'fast'
            elif cycle_duration_sec > 1.4:
                cycle.cycle_type = 'slow'
            else:
                cycle.cycle_type = 'normal'
            
            # Analyze cycle features
            cycle.features.update({
                'duration_seconds': cycle_duration_sec,
                'similarity_variance': np.var(cycle.similarity_curve),
                'peak_prominence': np.max(cycle.similarity_curve) - np.min(cycle.similarity_curve),
                'smoothness': 1.0 / (1.0 + np.std(np.diff(cycle.similarity_curve))),
                'completeness': self._compute_cycle_completeness(cycle)
            })
        
        return cycles
    
    def _compute_cycle_completeness(self, cycle: GaitCycle) -> float:
        """Compute how complete a gait cycle is."""
        # A complete gait cycle should have:
        # 1. Clear peak (legs together)
        # 2. Clear troughs (legs apart)
        # 3. Smooth transitions
        
        curve = cycle.similarity_curve
        peak_idx = np.argmax(curve)
        trough_idx = np.argmin(curve)
        
        # Check if peak and trough are well-separated
        separation = abs(peak_idx - trough_idx) / len(curve)
        
        # Check for smooth transitions
        smoothness = 1.0 / (1.0 + np.std(np.diff(curve)))
        
        completeness = (separation * 0.5 + smoothness * 0.5)
        return min(1.0, completeness)
    
    def _compute_gait_metrics(self, cycles: List[GaitCycle], 
                            similarity_curve: np.ndarray,
                            frame_rate: float) -> GaitAnalysis:
        """Compute overall gait metrics."""
        if not cycles:
            return self._create_fallback_analysis([], "No cycles detected")
        
        # Compute average cycle duration
        durations = [cycle.duration for cycle in cycles]
        avg_duration = np.mean(durations)
        
        # Compute cycle consistency
        duration_std = np.std(durations)
        consistency = max(0, 1 - duration_std / avg_duration) if avg_duration > 0 else 0
        
        # Determine gait speed
        avg_duration_sec = avg_duration / frame_rate
        if avg_duration_sec < 0.9:
            gait_speed = 'fast'
        elif avg_duration_sec > 1.3:
            gait_speed = 'slow'
        else:
            gait_speed = 'normal'
        
        # Determine gait regularity
        if consistency > 0.8:
            gait_regularity = 'regular'
        elif consistency > 0.6:
            gait_regularity = 'slightly_irregular'
        else:
            gait_regularity = 'irregular'
        
        # Compute overall confidence
        avg_confidence = np.mean([cycle.confidence for cycle in cycles])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(cycles, consistency, gait_speed)
        
        # Create cycle boundaries
        cycle_boundaries = [(cycle.start_frame, cycle.end_frame) for cycle in cycles]
        
        return GaitAnalysis(
            total_frames=len(similarity_curve),
            detected_cycles=cycles,
            average_cycle_duration=avg_duration,
            cycle_consistency=consistency,
            gait_speed=gait_speed,
            gait_regularity=gait_regularity,
            similarity_curve=similarity_curve,
            cycle_boundaries=cycle_boundaries,
            analysis_confidence=avg_confidence,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, cycles: List[GaitCycle], 
                                consistency: float, gait_speed: str) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if len(cycles) < 2:
            recommendations.append("Collect more frames for better gait analysis")
        
        if consistency < 0.7:
            recommendations.append("Gait cycles are irregular - consider longer sequences")
        
        if gait_speed == 'fast':
            recommendations.append("Fast gait detected - may affect recognition accuracy")
        elif gait_speed == 'slow':
            recommendations.append("Slow gait detected - may indicate walking issues")
        
        # Check for cycle quality
        low_quality_cycles = [c for c in cycles if c.confidence < 0.6]
        if low_quality_cycles:
            recommendations.append(f"{len(low_quality_cycles)} low-quality cycles detected")
        
        if not recommendations:
            recommendations.append("Gait analysis quality is good")
        
        return recommendations
    
    def _create_fallback_analysis(self, silhouettes: List[np.ndarray], 
                                reason: str) -> GaitAnalysis:
        """Create fallback analysis when proper analysis fails."""
        return GaitAnalysis(
            total_frames=len(silhouettes),
            detected_cycles=[],
            average_cycle_duration=0,
            cycle_consistency=0,
            gait_speed='unknown',
            gait_regularity='unknown',
            similarity_curve=np.array([]),
            cycle_boundaries=[],
            analysis_confidence=0,
            recommendations=[f"Analysis failed: {reason}"]
        )
    
    def visualize_gait_analysis(self, analysis: GaitAnalysis, 
                              save_path: Optional[str] = None) -> None:
        """
        Visualize gait analysis results.
        
        Args:
            analysis: GaitAnalysis results
            save_path: Optional path to save visualization
        """
        if not analysis.detected_cycles:
            print("⚠️ No cycles to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gait Cycle Analysis Results', fontsize=16)
        
        # Plot 1: Similarity curve with cycle boundaries
        axes[0, 0].plot(analysis.similarity_curve, 'b-', label='Similarity Curve')
        
        # Mark cycle boundaries
        for start, end in analysis.cycle_boundaries:
            axes[0, 0].axvspan(start, end, alpha=0.3, color='red')
        
        axes[0, 0].set_title('Similarity Curve with Detected Cycles')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Similarity')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Cycle durations
        durations = [cycle.duration for cycle in analysis.detected_cycles]
        axes[0, 1].bar(range(len(durations)), durations)
        axes[0, 1].set_title('Cycle Durations')
        axes[0, 1].set_xlabel('Cycle Index')
        axes[0, 1].set_ylabel('Duration (frames)')
        axes[0, 1].grid(True)
        
        # Plot 3: Cycle confidence scores
        confidences = [cycle.confidence for cycle in analysis.detected_cycles]
        axes[1, 0].bar(range(len(confidences)), confidences, color='green')
        axes[1, 0].set_title('Cycle Confidence Scores')
        axes[1, 0].set_xlabel('Cycle Index')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].grid(True)
        
        # Plot 4: Cycle types
        cycle_types = [cycle.cycle_type for cycle in analysis.detected_cycles]
        type_counts = {}
        for cycle_type in cycle_types:
            type_counts[cycle_type] = type_counts.get(cycle_type, 0) + 1
        
        if type_counts:
            axes[1, 1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Cycle Types Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        
        plt.show()
    
    def export_analysis(self, analysis: GaitAnalysis, 
                       output_path: str) -> None:
        """
        Export gait analysis results to JSON file.
        
        Args:
            analysis: GaitAnalysis results
            output_path: Path to save JSON file
        """
        # Convert analysis to dictionary
        analysis_dict = {
            'total_frames': analysis.total_frames,
            'average_cycle_duration': analysis.average_cycle_duration,
            'cycle_consistency': analysis.cycle_consistency,
            'gait_speed': analysis.gait_speed,
            'gait_regularity': analysis.gait_regularity,
            'analysis_confidence': analysis.analysis_confidence,
            'recommendations': analysis.recommendations,
            'cycles': []
        }
        
        for cycle in analysis.detected_cycles:
            cycle_dict = {
                'start_frame': cycle.start_frame,
                'end_frame': cycle.end_frame,
                'duration': cycle.duration,
                'peak_frame': cycle.peak_frame,
                'trough_frames': cycle.trough_frames,
                'confidence': cycle.confidence,
                'cycle_type': cycle.cycle_type,
                'features': cycle.features
            }
            analysis_dict['cycles'].append(cycle_dict)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(analysis_dict, f, indent=2)
        
        print(f"✅ Analysis exported to {output_path}")

# Example usage
if __name__ == "__main__":
    print("🧪 Testing Gait Cycle Analyzer")
    
    # Create test data
    test_silhouettes = []
    for i in range(100):
        # Create a simple periodic pattern
        phase = (i * 2 * np.pi) / 30  # 30-frame cycle
        similarity = 0.5 + 0.3 * np.sin(phase) + 0.1 * np.random.randn()
        similarity = np.clip(similarity, 0, 1)
        
        # Create dummy silhouette
        silhouette = np.random.randint(0, 255, (64, 44), dtype=np.uint8)
        test_silhouettes.append(silhouette)
    
    # Analyze gait cycles
    analyzer = GaitCycleAnalyzer()
    analysis = analyzer.analyze_gait_cycles(test_silhouettes, frame_rate=30.0)
    
    print(f"📊 Analysis Results:")
    print(f"   Detected cycles: {len(analysis.detected_cycles)}")
    print(f"   Average duration: {analysis.average_cycle_duration:.1f} frames")
    print(f"   Consistency: {analysis.cycle_consistency:.3f}")
    print(f"   Gait speed: {analysis.gait_speed}")
    print(f"   Regularity: {analysis.gait_regularity}")
    print(f"   Confidence: {analysis.analysis_confidence:.3f}")
    
    # Visualize results
    analyzer.visualize_gait_analysis(analysis) 