# """
# Script to run the fraud detection API server.
# """

# import uvicorn
# import os
# import sys
# from pathlib import Path

# # Add project root to Python path
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# from src.fraud_detector.api.config import settings

# if __name__ == "__main__":
#     uvicorn.run(
#         "src.fraud_detector.api.main:app",
#         host=settings.host,
#         port=settings.port,
#         reload=settings.reload,
#         log_level=settings.log_level.lower(),
#         access_log=True
#     )
