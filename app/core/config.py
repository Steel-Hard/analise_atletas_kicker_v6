from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Análise Atletas API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # MongoDB Setup
    MONGODB_URI: str = "mongodb://localhost:27017/"
    DATABASE_NAME: str = "analise_atletas"

    # Security
    SECRET_KEY: str = "your-super-secret-key-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440 # 24 hours

    class Config:
        env_file = ".env"

settings = Settings()
