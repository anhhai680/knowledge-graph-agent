#!/usr/bin/env python3
"""
Simple development server for the Knowledge Graph Agent web interface.
Run this instead of Docker for faster development with live reloading.
"""

import http.server
import socketserver
import os
import sys
import argparse
from pathlib import Path

# Change to the web directory
web_dir = Path(__file__).parent
os.chdir(web_dir)

class DevHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve index.html for all routes (SPA behavior)"""
    
    def end_headers(self):
        # Add headers to prevent caching during development
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def do_GET(self):
        # Serve index.html for all requests (except for assets)
        if self.path == '/' or not '.' in self.path.split('/')[-1]:
            self.path = '/index.html'
        return super().do_GET()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Development server for Knowledge Graph Agent')
    parser.add_argument('--port', type=int, default=3000, help='Port to run the server on (default: 3000)')
    args = parser.parse_args()
    
    PORT = args.port
    
    print(f"Starting development server on http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        with socketserver.TCPServer(("", PORT), DevHandler) as httpd:
            print(f"Serving at http://localhost:{PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down the development server...")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Error: Port {PORT} is already in use.")
            print("Please stop the existing server or use a different port.")
        else:
            print(f"Error starting server: {e}")
        sys.exit(1)
