import os
from pydantic import BaseModel

class Settings(BaseModel):
    APP_NAME: str = "UK PII Detector & Redactor"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    MAX_BATCH_SIZE: int = 100
    MAX_TEXT_LENGTH: int = 50000

settings = Settings()
