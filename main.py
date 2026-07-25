#!/usr/bin/env python3
import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import and run the server
from dashboard.server import app
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8642))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
