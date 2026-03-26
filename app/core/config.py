# from pydantic_settings import BaseSettings

# class Settings(BaseSettings):
#     DB_HOST: str = "127.0.0.1"
#     DB_PORT: int = 3306
#     DB_NAME: str = "fish_price_db"
#     DB_USER: str = "root"
#     DB_PASSWORD: str = ""

#     JWT_SECRET_KEY: str = "change_me"
#     JWT_ALGORITHM: str = "HS256"
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

#     CORS_ORIGINS: str = "http://localhost:3000"

#     class Config:
#         env_file = ".env"

# settings = Settings()


# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
# MODEL_DIR = os.path.join(BASE_DIR, "models_store")

# # Ensure folders exist
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(MODEL_DIR, exist_ok=True)


from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # Database
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_NAME: str = "fish_price_db"
    DB_USER: str = "root"
    DB_PASSWORD: str = ""

    # Optional full database URL for Railway
    DATABASE_URL: str | None = None

    # JWT
    JWT_SECRET_KEY: str = "change_me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    # Frontend URL
    CORS_ORIGINS: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models_store")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)