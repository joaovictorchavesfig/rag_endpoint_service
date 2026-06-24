from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    supabase_url: str
    supabase_service_key: str
    port: int = 8000

    # OpenAI models — cheapest options
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Vector search defaults
    match_count: int = 5
    match_threshold: float = 0.5


settings = Settings()
