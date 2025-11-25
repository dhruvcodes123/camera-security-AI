import torch
import torch.nn as nn
import cv2
import numpy as np
import yaml
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
from typing import Tuple, Optional
import sys
import os
from pathlib import Path

# Add OpenGait to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'external' / 'opengait'))

# Import RVM-only silhouette extractor
try:
    from gait_extraction.rvm_only_silhouette import create_rvm_only_silhouette_extractor
    RVM_ONLY_AVAILABLE = True
except ImportError:
    RVM_ONLY_AVAILABLE = False

# Import silhouette configuration
try:
    from silhouette_config import should_use_rvm, get_silhouette_method
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Import real SwinGait model
from external.opengait.opengait.modeling.swingait import SwinGait

# Import gait cycle analyzer
from gait_extraction.gait_cycle_analyzer import GaitCycleAnalyzer, GaitAnalysis

class GaitExtractor:
    def __init__(self,
                 config_file="external/opengait/configs/swingait/swingait_casia-b.yaml",
                 weights_path="models/gait/swingait_pretrained.pth"):  # Changed to working model
        
        # Load config
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Initialize real SwinGait model
        self.model = SwinGait(config, training=False)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.cuda()

        self.resolution = (128, 88)  # height, width
        
        # Initialize RVM silhouette extractor
        if RVM_ONLY_AVAILABLE:
            self.silhouette_extractor = create_rvm_only_silhouette_extractor()
            # print("✅ RVM-only silhouette extractor initialized")
        else:
            raise ImportError("RVM silhouette extractor not available. Please ensure RVM is properly installed.")
        
        # Initialize gait cycle analyzer
        self.gait_cycle_analyzer = GaitCycleAnalyzer()
        # print("✅ Gait cycle analyzer initialized")

        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            self.model.load_state_dict(state_dict['model'], strict=False)
            # print("✅ Real SwinGait model loaded successfully with pre-trained weights.")
        except FileNotFoundError:
            print(f"❌ ERROR: SwinGait weights not found at {weights_path}.")
            print("   Please ensure the model file exists and is not corrupted.")
            print("   Using random initialization - performance will be limited.")
        except Exception as e:
            print(f"❌ ERROR: Failed to load SwinGait model from {weights_path}: {e}")
            print("   The model file may be corrupted or incompatible.")
            print("   Using random initialization - performance will be limited.")

    def _create_silhouette(self, image: np.ndarray, pose_results=None) -> np.ndarray:
        """
        Create silhouette mask from an RGB image using RVM silhouette extractor.
        """
        # Use the RVM extract_silhouette method
        if hasattr(self.silhouette_extractor, 'extract_silhouette'):
            silhouette = self.silhouette_extractor.extract_silhouette(image)
        else:
            raise RuntimeError("RVM silhouette extractor does not have extract_silhouette method")
        return silhouette

    def extract_embedding(self, image_sequence: list) -> Optional[Tuple[np.ndarray, str, list]]:
        if not image_sequence:
            return None

        # --- Frame processing logic ---
        # Using all consecutive frames as requested. Frame skipping is disabled.
        print(f"Processing all {len(image_sequence)} consecutive frames for gait analysis.")
        
        silhouettes = []
        frontal_view_count = 0
        back_view_count = 0
        
        for i, image in enumerate(image_sequence):
            silhouette = self._create_silhouette(image)
            resized_silhouette = cv2.resize(silhouette, (self.resolution[1], self.resolution[0]))
            silhouettes.append(resized_silhouette)

        if frontal_view_count > back_view_count:
            view_type = 'frontal'
        else:
            view_type = 'back'
            
        # --- 2. Compute Silhouette Similarity Curve ---
        gait_cycle_frames = silhouettes  # Default to all frames
        gait_cycle_found = False
        
        if len(silhouettes) < 30: # Need at least 30 frames for reliable 2-cycle detection
            print(f"Warning: Not enough frames ({len(silhouettes)}) to detect 2 full gait cycles. Using all frames.")
            gait_cycle_frames = silhouettes
            gait_cycle_found = False
            start_index = 0
            end_index = len(silhouettes)
        else:
            flat_silhouettes = [s.flatten().reshape(1, -1) for s in silhouettes]
            
            similarities = []
            # Compare each frame's silhouette to the very first frame's silhouette
            # This ensures the `similarities` array has the same length as `silhouettes`
            for i in range(len(flat_silhouettes)):
                sim = cosine_similarity(flat_silhouettes[i], flat_silhouettes[0])
                similarities.append(sim[0][0])
            
            # --- 3. Detect Gait Cycle using Trough-Peak-Trough ---
            # A gait cycle is defined as a sequence from one leg-apart stance to the next,
            # passing through a legs-together stance. This corresponds to a Trough -> Peak -> Trough
            # sequence in the similarity curve.

            # Find peaks (high similarity, legs together)
            # A peak must have a similarity value of at least 0.4 (lowered for better detection)
            peaks, _ = find_peaks(similarities, distance=12, height=0.4)
            
            # Find troughs (low similarity, legs apart) by finding peaks in the inverted signal
            inverted_similarities = -np.array(similarities)
            troughs, _ = find_peaks(inverted_similarities, distance=12)

            # --- MODIFIED: Find exactly 2 gait cycles, get embedding for each, then average ---
            individual_embeddings = []
            all_cycle_frames = []

            if len(troughs) >= 2 and len(peaks) >= 1:
                print(f"Found {len(troughs)} troughs and {len(peaks)} peaks. Searching for exactly 2 gait cycles.")
                
                # We need at least two troughs to define a cycle.
                # A full cycle is from one trough to the next trough.
                
                cycles_found = 0
                for i in range(len(troughs) - 1):
                    start_trough_idx = troughs[i]
                    end_trough_idx = troughs[i+1]
                    
                    # Check if there is a peak between the troughs
                    has_peak_in_between = any(start_trough_idx < p < end_trough_idx for p in peaks)
                    
                    if has_peak_in_between:
                        print(f"Processing cycle {cycles_found + 1} from frame {start_trough_idx} to {end_trough_idx}.")
                        
                        # Extract frames for this specific cycle
                        cycle_silhouettes = silhouettes[start_trough_idx:end_trough_idx]
                        all_cycle_frames.extend(cycle_silhouettes)
                        
                        # Get embedding for this single cycle
                        cycle_tensor = torch.tensor(np.array(cycle_silhouettes), dtype=torch.float32)
                        cycle_tensor = cycle_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
                        if torch.cuda.is_available():
                            cycle_tensor = cycle_tensor.cuda()
                        
                        with torch.no_grad():
                            # Create dummy inputs for SwinGait model
                            dummy_labs = torch.zeros(1, dtype=torch.long)
                            # seqL should be [1, batch_size] format for PackSequenceWrapper
                            seqL = torch.tensor([[len(cycle_silhouettes)]], dtype=torch.long)  # [1, 1] format
                            
                            # Remove the channel dimension to make it 4D: (batch, sequence, height, width)
                            cycle_tensor_4d = cycle_tensor.squeeze(1)  # Remove channel dim
                            model_inputs = ([cycle_tensor_4d], dummy_labs, None, None, seqL)
                            
                            # Get embeddings from SwinGait
                            model_output = self.model(model_inputs)
                            embedding = model_output['inference_feat']['embeddings']
                            
                            # Normalize individual cycle embeddings
                            embedding_np = embedding.cpu().numpy().flatten()
                            embedding_norm = np.linalg.norm(embedding_np)
                            if embedding_norm > 0:
                                embedding_np = embedding_np / embedding_norm
                            individual_embeddings.append(embedding_np)
                        
                        cycles_found += 1
                        if cycles_found == 2:
                            break # We only need two cycles
                
                if individual_embeddings:
                    print(f"Averaging {len(individual_embeddings)} individual gait embeddings.")
                    avg_embedding = np.mean(individual_embeddings, axis=0)
                    gait_cycle_found = True
                    gait_cycle_frames = all_cycle_frames
                else:
                    print("Could not isolate 2 full gait cycles. Using all frames as fallback.")
                    gait_cycle_found = False

        if not gait_cycle_frames:
            print("No silhouettes generated, cannot extract embedding.")
            return None
        
        # --- 4. Final Embedding Generation ---
        final_silhouettes_np = np.array(gait_cycle_frames, dtype=np.float32)
        # Ensure correct tensor shape: (batch, channel, sequence, height, width)
        # SwinGait expects: (1, 1, num_frames, 64, 44)
        final_silhouettes_tensor = torch.tensor(final_silhouettes_np, dtype=torch.float32)
        final_silhouettes_tensor = final_silhouettes_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        print(f"DEBUG: Tensor shape: {final_silhouettes_tensor.shape}")
        
        if torch.cuda.is_available():
            final_silhouettes_tensor = final_silhouettes_tensor.cuda()

        with torch.no_grad():
            # Create dummy inputs for SwinGait model
            # SwinGait expects: (ipts, labs, _, _, seqL) where ipts is a list of tensors
            # The tensor should be 4D: (batch, sequence, height, width) - no channel dim
            dummy_labs = torch.zeros(1, dtype=torch.long)  # Dummy labels
            # seqL should be [1, batch_size] format for PackSequenceWrapper
            seqL = torch.tensor([[len(gait_cycle_frames)]], dtype=torch.long)  # [1, 1] format
            
            # Remove the channel dimension to make it 4D: (batch, sequence, height, width)
            final_silhouettes_4d = final_silhouettes_tensor.squeeze(1)  # Remove channel dim
            
            # Create the input format expected by SwinGait
            model_inputs = ([final_silhouettes_4d], dummy_labs, None, None, seqL)
            
            # Get embeddings from SwinGait
            model_output = self.model(model_inputs)
            final_embedding = model_output['inference_feat']['embeddings']
            
        # Normalize the gait embedding to unit length
        embedding_np = final_embedding.cpu().numpy().flatten()
        embedding_norm = np.linalg.norm(embedding_np)
        if embedding_norm > 0:
            embedding_np = embedding_np / embedding_norm
            
        print(f"✅ Real SwinGait extracted {len(embedding_np)}D embedding")
        return embedding_np, view_type, gait_cycle_frames
    
    def analyze_gait_cycles_comprehensive(self, image_sequence: list, 
                                        frame_rate: float = 30.0,
                                        save_analysis: bool = False,
                                        output_dir: str = "gait_analysis_results") -> Optional[GaitAnalysis]:
        """
        Perform comprehensive gait cycle analysis with detailed metrics.
        
        Args:
            image_sequence: List of RGB images
            frame_rate: Frame rate of the video (fps)
            save_analysis: Whether to save analysis results
            output_dir: Directory to save analysis results
            
        Returns:
            GaitAnalysis: Comprehensive gait analysis results
        """
        if not image_sequence:
            print("❌ No images provided for gait cycle analysis")
            return None
        
        print(f"🔍 Starting comprehensive gait cycle analysis for {len(image_sequence)} frames...")
        
        # Step 1: Extract silhouettes
        silhouettes = []
        for i, image in enumerate(image_sequence):
            silhouette = self._create_silhouette(image)
            resized_silhouette = cv2.resize(silhouette, (self.resolution[1], self.resolution[0]))
            silhouettes.append(resized_silhouette)
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(image_sequence)} frames")
        
        print(f"✅ Extracted {len(silhouettes)} silhouettes")
        
        # Step 2: Perform comprehensive gait cycle analysis
        gait_analysis = self.gait_cycle_analyzer.analyze_gait_cycles(silhouettes, frame_rate)
        
        # Step 3: Print analysis summary
        self._print_gait_analysis_summary(gait_analysis)
        
        # Step 4: Save analysis results if requested
        if save_analysis:
            self._save_gait_analysis_results(gait_analysis, output_dir)
        
        return gait_analysis
    
    def _print_gait_analysis_summary(self, analysis: GaitAnalysis) -> None:
        """Print a comprehensive summary of gait analysis results."""
        print("\n" + "="*60)
        print("🎯 GAIT CYCLE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📊 Total Frames Analyzed: {analysis.total_frames}")
        print(f"🔄 Detected Gait Cycles: {len(analysis.detected_cycles)}")
        print(f"⏱️  Average Cycle Duration: {analysis.average_cycle_duration:.1f} frames")
        print(f"📈 Cycle Consistency: {analysis.cycle_consistency:.3f}")
        print(f"🏃 Gait Speed: {analysis.gait_speed.upper()}")
        print(f"🔄 Gait Regularity: {analysis.gait_regularity.upper()}")
        print(f"🎯 Analysis Confidence: {analysis.analysis_confidence:.3f}")
        
        if analysis.detected_cycles:
            print(f"\n📋 CYCLE DETAILS:")
            for i, cycle in enumerate(analysis.detected_cycles):
                print(f"   Cycle {i+1}:")
                print(f"     • Frames: {cycle.start_frame}-{cycle.end_frame} ({cycle.duration} frames)")
                print(f"     • Type: {cycle.cycle_type}")
                print(f"     • Confidence: {cycle.confidence:.3f}")
                print(f"     • Peak Frame: {cycle.peak_frame}")
                print(f"     • Trough Frames: {cycle.trough_frames}")
        
        if analysis.recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in analysis.recommendations:
                print(f"   • {rec}")
        
        print("="*60)
    
    def _save_gait_analysis_results(self, analysis: GaitAnalysis, output_dir: str) -> None:
        """Save gait analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save analysis as JSON
        json_path = os.path.join(output_dir, "gait_analysis.json")
        self.gait_cycle_analyzer.export_analysis(analysis, json_path)
        
        # Save visualization
        viz_path = os.path.join(output_dir, "gait_analysis_visualization.png")
        self.gait_cycle_analyzer.visualize_gait_analysis(analysis, viz_path)
        
        # Save individual cycle silhouettes
        cycles_dir = os.path.join(output_dir, "cycles")
        os.makedirs(cycles_dir, exist_ok=True)
        
        for i, cycle in enumerate(analysis.detected_cycles):
            cycle_dir = os.path.join(cycles_dir, f"cycle_{i+1}")
            os.makedirs(cycle_dir, exist_ok=True)
            
            for j, silhouette in enumerate(cycle.silhouettes):
                frame_path = os.path.join(cycle_dir, f"frame_{j:03d}.png")
                cv2.imwrite(frame_path, silhouette)
        
        print(f"✅ Gait analysis results saved to {output_dir}")
    
    def extract_gait_embedding_with_cycle_analysis(self, image_sequence: list,
                                                 frame_rate: float = 30.0,
                                                 use_cycle_analysis: bool = True) -> Optional[Tuple[np.ndarray, str, list, GaitAnalysis]]:
        """
        Extract gait embedding with optional cycle analysis.
        
        Args:
            image_sequence: List of RGB images
            frame_rate: Frame rate of the video (fps)
            use_cycle_analysis: Whether to use cycle analysis for frame selection
            
        Returns:
            Tuple containing: (embedding, view_type, selected_frames, gait_analysis)
        """
        if not image_sequence:
            return None
        
        print(f"🔍 Extracting gait embedding with cycle analysis...")
        
        # Perform gait cycle analysis
        gait_analysis = self.gait_cycle_analyzer.analyze_gait_cycles(image_sequence, frame_rate)
        
        if use_cycle_analysis and gait_analysis.detected_cycles:
            # Use cycle analysis to select optimal frames
            selected_frames = self._select_frames_based_on_cycles(gait_analysis, image_sequence)
            print(f"✅ Selected {len(selected_frames)} frames based on cycle analysis")
        else:
            # Use all frames (fallback)
            selected_frames = image_sequence
            print(f"⚠️ Using all {len(selected_frames)} frames (no cycle analysis)")
        
        # Extract embedding using selected frames
        result = self.extract_embedding(selected_frames)
        
        if result:
            embedding, view_type, processed_frames = result
            return embedding, view_type, processed_frames, gait_analysis
        else:
            return None
    
    def _select_frames_based_on_cycles(self, analysis: GaitAnalysis, 
                                     image_sequence: list) -> list:
        """Select optimal frames based on gait cycle analysis."""
        selected_frames = []
        
        # Select frames from the best cycles
        best_cycles = sorted(analysis.detected_cycles, 
                           key=lambda x: x.confidence, reverse=True)
        
        for cycle in best_cycles[:3]:  # Use top 3 cycles
            cycle_frames = image_sequence[cycle.start_frame:cycle.end_frame + 1]
            selected_frames.extend(cycle_frames)
        
        # If we don't have enough frames, add some from other cycles
        if len(selected_frames) < 30:
            remaining_cycles = best_cycles[3:]
            for cycle in remaining_cycles:
                if len(selected_frames) >= 30:
                    break
                cycle_frames = image_sequence[cycle.start_frame:cycle.end_frame + 1]
                selected_frames.extend(cycle_frames)
        
        # Ensure we have at least some frames
        if not selected_frames:
            selected_frames = image_sequence
        
        return selected_frames

if __name__ == '__main__':
    # This part is for standalone testing of the extractor
    pass 