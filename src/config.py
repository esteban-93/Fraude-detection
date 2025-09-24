"""
API configuration settings.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
# from pydantic import BaseModel, Field, ConfigDict


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # API Settings
    title: str = "Fraud Detection API"
    description: str = "Real-time fraud detection for credit card transactions"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Model Settings
    model_path: str = "models/trained/fraud_model.pkl"
    scaler_path: str = "models/trained/scaler.pkl"
    model_version: str = "1.0.0"
    
    # Security Settings
    cors_origins: list = ["*"]
    rate_limit_calls_per_minute: int = 60
    
    # Logging Settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Database Settings (for future use)
    database_url: Optional[str] = None


    # model_config = ConfigDict(env_file=".env", case_sensitive = False)
    model_config = SettingsConfigDict(env_file=".env",
                                      env_prefix="FRAUD_API_",
                                      case_sensitive=False)
    
    # class Config:
    #     env_file = ".env"
    #     env_prefix = "FRAUD_API_"
    #     case_sensitive = False

# Global settings instance
settings = APISettings()
