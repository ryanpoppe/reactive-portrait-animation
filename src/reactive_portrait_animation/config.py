from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["mock", "openai", "anthropic", "elevenlabs", "ollama", "vllm", "local"]
Device = Literal["auto", "cpu", "cuda", "mps"]
Platform = Literal["windows", "linux", "macos"]
Environment = Literal["development", "test", "production"]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: Environment = Field(default="development", alias="RPA_ENV")
    log_level: str = Field(default="INFO", alias="RPA_LOG_LEVEL")

    persona_path: Path = Field(
        default=Path("configs/personas/default.toml"), alias="RPA_PERSONA_PATH"
    )
    data_dir: Path = Field(default=Path("data"), alias="RPA_DATA_DIR")
    cache_dir: Path = Field(default=Path(".cache/rpa"), alias="RPA_CACHE_DIR")

    device: Device = Field(default="auto", alias="RPA_DEVICE")
    accelerator: Device = Field(default="auto", alias="RPA_ACCELERATOR")
    platform: Platform = Field(default="windows", alias="RPA_PLATFORM")

    asr_provider: Provider = Field(default="mock", alias="RPA_ASR_PROVIDER")
    vision_provider: Provider = Field(default="mock", alias="RPA_VISION_PROVIDER")
    llm_provider: Provider = Field(default="mock", alias="RPA_LLM_PROVIDER")
    tts_provider: Provider = Field(default="mock", alias="RPA_TTS_PROVIDER")
    animation_provider: Provider = Field(default="mock", alias="RPA_ANIMATION_PROVIDER")

    llm_model: str = Field(default="gpt-4o-mini", alias="RPA_LLM_MODEL")
    asr_model: str = Field(default="whisper-small", alias="RPA_ASR_MODEL")
    tts_model: str = Field(default="eleven_multilingual_v2", alias="RPA_TTS_MODEL")
    vision_model: str = Field(default="gpt-4o-mini", alias="RPA_VISION_MODEL")
    animation_model: str = Field(default="liveportrait", alias="RPA_ANIMATION_MODEL")

    sam_checkpoint: Path = Field(
        default=Path("models/sam2.1_hiera_base_plus.pt"), alias="RPA_SAM_CHECKPOINT"
    )
    sam_model_cfg: str = Field(
        default="configs/sam2.1/sam2.1_hiera_b+.yaml", alias="RPA_SAM_MODEL_CFG"
    )
    feather_radius: int = Field(default=10, alias="RPA_FEATHER_RADIUS")

    target_fps: int = Field(default=30, alias="RPA_TARGET_FPS")
    max_latency_ms: int = Field(default=500, alias="RPA_MAX_LATENCY_MS")
    camera_device: str = Field(default="0", alias="RPA_CAMERA_DEVICE")
    mic_device: str = Field(default="default", alias="RPA_MIC_DEVICE")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    elevenlabs_api_key: str = Field(default="", alias="ELEVENLABS_API_KEY")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    vllm_base_url: str = Field(default="http://localhost:8000", alias="VLLM_BASE_URL")
    whisper_server_url: str = Field(default="http://localhost:9000", alias="WHISPER_SERVER_URL")
    tts_server_url: str = Field(default="http://localhost:5002", alias="TTS_SERVER_URL")
    animation_server_url: str = Field(default="http://localhost:7000", alias="ANIMATION_SERVER_URL")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
