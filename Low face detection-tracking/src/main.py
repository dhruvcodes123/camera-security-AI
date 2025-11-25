#!/usr/bin/env python
"""
Main entry point for the Hybrid Person Tracking System.
"""

import os
import sys
import warnings
from src.utils.logging import setup_logging
from .config import parse_args
from .core import HybridPersonTrackingApp


def main():
    """
    Main function to run the Hybrid Person Tracking System.
    """
    # Ignore warnings
    warnings.filterwarnings("ignore")
    
    # Setup logging
    setup_logging(log_to_file=True)
    
    # Parse command line arguments
    args = parse_args()
    
    # Create and initialize app
    app = HybridPersonTrackingApp(args)
    if not app.initialize():
        sys.exit(1)
    
    # Run app
    app.run()


if __name__ == "__main__":
    main() 