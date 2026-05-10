from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_seed_lesson_uses_env_and_admin_header() -> None:
    script = (ROOT / "scripts" / "seed_lesson.py").read_text(encoding="utf-8")

    assert "load_dotenv()" in script
    assert "ADMIN_API_KEY" in script
    assert "X-Admin-API-Key" in script
    assert "Aula 1: Verb To Be" in script
    assert "http://localhost:8000/api/curriculum/lesson" in script
    assert "gsk_" not in script


def test_set_webhook_uses_env_and_ngrok_input() -> None:
    script = (ROOT / "scripts" / "set_webhook.py").read_text(encoding="utf-8")

    assert "load_dotenv()" in script
    assert "EVOLUTION_API_URL" in script
    assert "EVOLUTION_API_KEY" in script
    assert "input(" in script
    assert "messages.upsert" in script
    assert "{ngrok_url}/webhook" in script
    assert "gsk_" not in script


def test_testing_guide_has_e2e_steps() -> None:
    guide = (ROOT / "TESTING_GUIDE.md").read_text(encoding="utf-8")

    assert "ngrok http 8000" in guide
    assert "python scripts/seed_lesson.py" in guide
    assert "python scripts/set_webhook.py" in guide
    assert "WhatsApp" in guide
