from app.config.settings import Settings
from app.database.vector_store import EMBEDDING_DIMENSION, HUGGINGFACE_MODEL_NAME


def test_settings_exposes_groq_and_edge_tts_without_paid_provider_keys() -> None:
    settings = Settings(GROQ_API_KEY="groq-key")

    assert settings.groq_api_key == "groq-key"
    assert settings.edge_tts_voice == "en-US-AriaNeural"
    assert not hasattr(settings, "openai_api_key")
    assert not hasattr(settings, "elevenlabs_api_key")


def test_open_source_embeddings_match_pgvector_dimension() -> None:
    assert HUGGINGFACE_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"
    assert EMBEDDING_DIMENSION == 384
