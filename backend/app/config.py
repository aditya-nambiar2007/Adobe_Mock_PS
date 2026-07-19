from __future__ import annotations

from pathlib import Path
from typing import Dict

from pydantic_settings import BaseSettings

from app.utils.gpu import cuda_available

from pydantic import SecretStr

class Settings(BaseSettings):
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.1-flash-lite"
    device: str = ""
    model_paths: Dict[str, str] = {}
    diffusion_lora_weights: str = ""
    diffusion_lora_adapter_name: str = "default"
    ghibli_lora_weights: str = ""
    ghibli_lora_adapter_name: str = "ghibli"
    output_directory: str = "outputs"
    upload_directory: str = "uploads"
    temp_directory: str = "temp"
    max_upload_size_mb: int = 10
    request_timeout_seconds: int = 300
    model_cache_timeout_minutes: int = 30
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173"

    @property
    def resolved_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if cuda_available() else "cpu"

    @property
    def is_gemini_configured(self) -> bool:
        return bool(self.gemini_api_key.get_secret_value())

    @property
    def output_path(self) -> Path:
        return Path(self.output_directory).resolve()

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_directory).resolve()

    @property
    def temp_path(self) -> Path:
        return Path(self.temp_directory).resolve()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
