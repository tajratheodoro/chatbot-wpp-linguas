"""Configure Evolution API webhook to point to the local FastAPI tunnel."""

from __future__ import annotations

import os
import sys
from urllib.parse import urljoin

import httpx
from dotenv import load_dotenv

DEFAULT_WEBHOOK_ENDPOINT = "/webhook/set"
TIMEOUT_SECONDS = 20.0


def normalize_url(url: str) -> str:
    """Normalize a developer-provided public URL without trailing slash."""
    normalized = url.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("a URL do Ngrok deve comecar com http:// ou https://")
    return normalized


def build_payload(ngrok_url: str) -> dict[str, object]:
    """Build Evolution API payload for messages.upsert webhook events."""
    return {
        "webhook": {
            "enabled": True,
            "url": f"{ngrok_url}/webhook",
            "webhook_by_events": True,
            "webhook_base64": True,
            "events": ["messages.upsert"],
        }
    }


def main() -> int:
    """Ask for an Ngrok URL and configure the Evolution API webhook."""
    load_dotenv()

    evolution_api_url = os.getenv("EVOLUTION_API_URL", "").strip().rstrip("/")
    evolution_api_key = os.getenv("EVOLUTION_API_KEY", "").strip()
    webhook_endpoint = os.getenv("EVOLUTION_WEBHOOK_ENDPOINT", DEFAULT_WEBHOOK_ENDPOINT).strip()

    if not evolution_api_url:
        print("Erro: EVOLUTION_API_URL nao encontrado no .env.", file=sys.stderr)
        return 1
    if not evolution_api_key:
        print("Erro: EVOLUTION_API_KEY nao encontrado no .env.", file=sys.stderr)
        return 1

    try:
        ngrok_url = normalize_url(input("Cole aqui sua URL do Ngrok: "))
    except ValueError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1

    target_url = urljoin(f"{evolution_api_url}/", webhook_endpoint.lstrip("/"))
    headers = {"apikey": evolution_api_key}
    try:
        response = httpx.post(
            target_url,
            json=build_payload(ngrok_url),
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        print("Erro: Evolution API offline ou inacessivel.", file=sys.stderr)
        return 1
    except httpx.TimeoutException:
        print("Erro: timeout ao conectar na Evolution API.", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        print(
            f"Erro: Evolution API retornou status {exc.response.status_code}: "
            f"{exc.response.text}",
            file=sys.stderr,
        )
        return 1
    except httpx.RequestError as exc:
        print(f"Erro de conexao ao configurar webhook: {exc}", file=sys.stderr)
        return 1

    print(f"Webhook configurado com sucesso para {ngrok_url}/webhook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
