from app.core.config import get_settings


def test_default_settings_load():
    settings = get_settings()
    assert settings.app.name == "advanced-rag-engine"
    assert settings.app.port == 5000
    assert settings.logging.json is True
