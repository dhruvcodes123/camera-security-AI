#!/usr/bin/env python3
"""
Silhouette Extraction Configuration
Configure which silhouette extraction method to use: RVM
"""

# =============================================================================
# SILHOUETTE EXTRACTION CONFIGURATION
# =============================================================================

# Toggle between different silhouette extraction methods
SILHOUETTE_EXTRACTION_METHOD = {
    # Primary method for silhouette extraction
    'method': 'RVM',  # Options: 'RVM' (MOG2 removed)
    
    # RVM-specific settings
    'rvm': {
        'enabled': True,  # Set to True to enable RVM
        'model_path': 'RobustVideoMatting/pretrained/rvm_mobilenetv3.pth',
        'device': 'auto',  # 'auto', 'cpu', 'cuda'
        'downsample_ratio': 0.5,  # Better quality (reduced from 0.25)
        'threshold': 0.3,  # Binary threshold for alpha matte (enhanced quality)
        'save_silhouettes': True,  # Save to gait_cycles folder
    },
    
    # Performance settings
    'performance': {
        'batch_processing': False,  # Process frames in batches
        'temporal_consistency': True,  # Use temporal consistency (RVM)
        'parallel_processing': False,  # Multi-threading (future feature)
    }
}

# =============================================================================
# SPEED COMPARISON SETTINGS
# =============================================================================

# For testing and comparison
SPEED_COMPARISON = {
    'enabled': False,  # Set to True to compare speeds
    'methods_to_test': ['RVM'],  # Removed MOG2
    'test_frames': 50,  # Number of frames for speed test
    'output_results': True,  # Print speed comparison results
}

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def get_silhouette_method():
    """Get the current silhouette extraction method."""
    return SILHOUETTE_EXTRACTION_METHOD['method']

def is_rvm_enabled():
    """Check if RVM is enabled."""
    return SILHOUETTE_EXTRACTION_METHOD['rvm']['enabled']

def should_use_rvm():
    """Check if RVM should be used."""
    return SILHOUETTE_EXTRACTION_METHOD['rvm']['enabled']

def get_rvm_config():
    """Get RVM configuration."""
    return SILHOUETTE_EXTRACTION_METHOD['rvm']

def get_performance_config():
    """Get performance configuration."""
    return SILHOUETTE_EXTRACTION_METHOD['performance']

def print_current_config():
    """Print current silhouette extraction configuration."""
    print("🔧 Current Silhouette Extraction Configuration:")
    print(f"   Method: {get_silhouette_method()}")
    print(f"   RVM Enabled: {is_rvm_enabled()}")
    
    if is_rvm_enabled():
        rvm_config = get_rvm_config()
        print(f"   RVM Model Path: {rvm_config['model_path']}")
        print(f"   RVM Device: {rvm_config['device']}")
        print(f"   RVM Downsample Ratio: {rvm_config['downsample_ratio']}")
        print(f"   RVM Threshold: {rvm_config['threshold']}")
        print(f"   Save Silhouettes: {rvm_config['save_silhouettes']}")
    
    perf_config = get_performance_config()
    print(f"   Batch Processing: {perf_config['batch_processing']}")
    print(f"   Temporal Consistency: {perf_config['temporal_consistency']}")
    print(f"   Parallel Processing: {perf_config['parallel_processing']}")

def update_silhouette_config(method='RVM', **kwargs):
    """Update silhouette extraction configuration."""
    global SILHOUETTE_EXTRACTION_METHOD
    
    if method == 'RVM':
        SILHOUETTE_EXTRACTION_METHOD['method'] = 'RVM'
        if 'rvm' in kwargs:
            SILHOUETTE_EXTRACTION_METHOD['rvm'].update(kwargs['rvm'])
    
    if 'performance' in kwargs:
        SILHOUETTE_EXTRACTION_METHOD['performance'].update(kwargs['performance'])

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_configuration():
    """Test the current configuration."""
    print("🧪 Testing Silhouette Configuration...")
    
    # Print current config
    print_current_config()
    
    # Test method selection
    method = get_silhouette_method()
    print(f"\n✅ Selected method: {method}")
    
    # Test RVM configuration
    if method == 'RVM':
        if is_rvm_enabled():
            print("✅ RVM is enabled")
            rvm_config = get_rvm_config()
            print(f"✅ RVM config loaded: {rvm_config['model_path']}")
        else:
            print("❌ RVM is disabled")
    
    print("\n✅ Configuration test completed!")

if __name__ == "__main__":
    test_configuration() 