from app.config.settings import Settings
from app.database.vector_store import DeterministicHashEmbeddings


def test_settings_exposes_groq_and_edge_tts_without_paid_provider_keys() -> None:
    settings = Settings(GROQ_API_KEY="groq-key")

    assert settings.groq_api_key == "groq-key"
    assert settings.edge_tts_voice == "en-US-AriaNeural"
    assert not hasattr(settings, "openai_api_key")
    assert not hasattr(settings, "elevenlabs_api_key")


def test_local_embedding_fallback_matches_pgvector_dimension() -> None:
    embeddings = DeterministicHashEmbeddings()

    vector = embeddings.embed_query("english lesson")

    assert len(vector) == 1536
    assert any(value != 0.0 for value in vector)
