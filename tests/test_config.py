from reactive_portrait_animation.config import AppSettings


def test_settings_defaults() -> None:
    settings = AppSettings()

    assert settings.environment == "development"
    assert settings.llm_provider == "mock"
    assert settings.animation_model == "liveportrait"
    assert settings.persona_path.parts[-3:] == ("configs", "personas", "default.toml")
