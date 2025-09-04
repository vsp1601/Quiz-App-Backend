from pydantic_settings import BaseSettings  # Changed from pydantic_settings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "sqlite:///./fashion_db.sqlite"  # Changed to SQLite for simplicity
    
    # Redis for caching
    REDIS_URL: str = "redis://localhost:6379"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # File paths
    IMAGES_PATH: str = "./images"
    DATA_PATH: str = "./data"
    MODELS_PATH: str = "./data/models"
    
    # ML Settings
    MIN_RATINGS_FOR_TRAINING: int = 10
    RECOMMENDATION_COUNT: int = 20
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Fashion Recommendation API"
    
    class Config:
        env_file = ".env"

settings = Settings()