"""Application settings and configuration management."""

import os
from functools import lru_cache
from typing import Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        # Fallback for older versions
        from pydantic import BaseModel as BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY_1")
    langchain_api_key: Optional[str] = Field(None, env="LANGCHAIN_API_KEY")
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = Field(True, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field("rm-agentic-ai", env="LANGCHAIN_PROJECT")
    
    # Application Settings
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_monitoring: bool = Field(True, env="ENABLE_MONITORING")
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    
    # Performance Settings
    max_concurrent_agents: int = Field(5, env="MAX_CONCURRENT_AGENTS")
    agent_timeout: int = Field(300, env="AGENT_TIMEOUT")
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    
    # File Paths
    data_dir: str = Field("data", env="DATA_DIR")
    models_dir: str = Field("models", env="MODELS_DIR")
    output_dir: str = Field("output", env="OUTPUT_DIR")
    
    # Model Settings
    risk_model_path: str = Field("models/risk_profile_model.pkl")
    goal_model_path: str = Field("models/goal_success_model.pkl")
    risk_encoders_path: str = Field("models/label_encoders.pkl")
    goal_encoders_path: str = Field("models/goal_success_label_encoders.pkl")
    
    # Data Files
    prospects_csv: str = Field("data/input_data/prospects.csv")
    products_csv: str = Field("data/input_data/products.csv")
    
    # Streamlit Configuration
    page_title: str = Field("AI-Powered Investment Analyzer")
    page_icon: str = Field("🤖")
    layout: str = Field("wide")
    
    # Agent Configuration
    default_temperature: float = Field(0.1)
    max_tokens: int = Field(4000)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()