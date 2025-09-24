"""
Entry point to run the fraud detection API server.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the API
from src.fraud_detector.api.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.fraud_detector.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
