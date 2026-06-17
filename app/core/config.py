from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "AnÃ¡lise Atletas API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # MongoDB Setup
    MONGODB_URI: str = "mongodb://localhost:27017/"
    DATABASE_NAME: str = "analise_atletas"

    # Security
    SECRET_KEY: str = "your-super-secret-key-change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440 # 24 hours

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
