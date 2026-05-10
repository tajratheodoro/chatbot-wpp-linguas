"""Seed an initial English lesson into the local FastAPI RAG curriculum."""

from __future__ import annotations

import os
import sys
from typing import Any

import httpx
from dotenv import load_dotenv

DEFAULT_CURRICULUM_URL = "http://localhost:8000/api/curriculum/lesson"
TIMEOUT_SECONDS = 20.0


def build_lesson_payload() -> dict[str, str]:
    """Return the structured payload for Aula 1: Verb To Be."""
    return {
        "title": "Aula 1: Verb To Be",
        "content": (
            "Aula 1: Verb To Be\n\n"
            "Objetivo pedagogico:\n"
            "- Ensinar o uso correto de am, is e are no presente simples.\n"
            "- Corrigir concordancia entre sujeito e verbo em frases basicas.\n\n"
            "Exercicio:\n"
            "Corrija as frases abaixo e explique brevemente o erro:\n"
            "1. I is a student.\n"
            "2. She are happy.\n"
            "3. They am from Brazil.\n"
            "4. He are my teacher.\n\n"
            "Gabarito:\n"
            "1. I am a student.\n"
            "2. She is happy.\n"
            "3. They are from Brazil.\n"
            "4. He is my teacher.\n\n"
            "Diretrizes pedagogicas para a IA:\n"
            "- Corrija com rigor, mas mantenha tom claro e encorajador.\n"
            "- Explique que I usa am, he/she/it usa is e you/we/they usa are.\n"
            "- Aponte apenas os erros relacionados ao Verb To Be quando possivel.\n"
            "- Responda em portugues brasileiro, mantendo os exemplos em ingles.\n"
        ),
    }


def main() -> int:
    """Send the seed lesson to the local curriculum admin endpoint."""
    load_dotenv()

    curriculum_url = os.getenv("CURRICULUM_ENDPOINT_URL", DEFAULT_CURRICULUM_URL).strip()
    admin_api_key = os.getenv("ADMIN_API_KEY", "").strip()
    if not admin_api_key:
        print("Erro: ADMIN_API_KEY nao encontrado no .env.", file=sys.stderr)
        return 1

    headers = {"X-Admin-API-Key": admin_api_key}
    try:
        response = httpx.post(
            curriculum_url,
            json=build_lesson_payload(),
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.ConnectError:
        print("Erro: FastAPI offline ou inacessivel em localhost:8000.", file=sys.stderr)
        return 1
    except httpx.TimeoutException:
        print("Erro: timeout ao conectar no FastAPI.", file=sys.stderr)
        return 1
    except httpx.HTTPStatusError as exc:
        detail: Any
        try:
            detail = exc.response.json()
        except ValueError:
            detail = exc.response.text
        print(
            f"Erro: seed falhou com status {exc.response.status_code}: {detail}",
            file=sys.stderr,
        )
        return 1
    except httpx.RequestError as exc:
        print(f"Erro de conexao ao enviar seed: {exc}", file=sys.stderr)
        return 1

    if response.status_code in {200, 201}:
        print("Aula 1: Verb To Be cadastrada com sucesso.")
        return 0

    print(f"Resposta inesperada do FastAPI: {response.status_code}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
