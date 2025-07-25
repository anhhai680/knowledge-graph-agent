#!/usr/bin/env python3
"""
Knowledge Graph Agent MVP - Main Application Entry Point.

This script initializes and runs the Knowledge Graph Agent application.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from src.api.routes import app
from src.utils.logging import setup_logging

# Load environment variables
load_dotenv()

def main():
    """Run the Knowledge Graph Agent application."""
    # Setup logging
    setup_logging()
    
    # Log application start
    logging.info("Starting Knowledge Graph Agent")
    
    # Import and run the application
    import uvicorn
    
    # Get the host and port from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    # Run the FastAPI application
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
