# config.py

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///./test.db"
    app_name: str = "My Data Platform"
    debug: bool = True
    port: int = 8000

def get_settings() -> Settings:
    return Settings()
  
