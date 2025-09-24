# """
# Configuration loader utility.
# """

# import yaml
# import os
# from typing import Dict, Any

# def load_config(config_path: str = "/Users/juan_esteban/Documents/data_science/Fraude-detection/configs/config.yaml") -> Dict[str, Any]:
#     """Load configuration from YAML file."""
#     try:
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
#         return config
#     except Exception as e:
#         print(f"Error loading config: {e}")
#         return {}

# def get_model_config() -> Dict[str, Any]:
#     """Get model-specific configuration."""
#     config = load_config()
#     return config.get('model', {})

# def get_training_config() -> Dict[str, Any]:
#     """Get training-specific configuration."""
#     config = load_config()
#     return config.get('training', {})
