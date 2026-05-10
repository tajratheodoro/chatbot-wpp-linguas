# Guia rapido de teste E2E

## 1. Subir API e banco

Com os containers ja configurados, deixe o FastAPI acessivel em `localhost:8000`.

```powershell
docker compose up --build
```

## 2. Abrir tunel publico com Ngrok

Em outro terminal, exponha a porta local da API:

```powershell
ngrok http 8000
```

Copie a URL publica HTTPS gerada pelo Ngrok.

## 3. Popular a primeira aula do RAG

Garanta que o `.env` tenha `ADMIN_API_KEY` igual ao valor usado pela API.

```powershell
python scripts/seed_lesson.py
```

O script cadastra a `Aula 1: Verb To Be` no endpoint de curriculo.

## 4. Configurar webhook da Evolution API

Garanta que o `.env` tenha:

```env
EVOLUTION_API_URL=...
EVOLUTION_API_KEY=...
EVOLUTION_WEBHOOK_ENDPOINT=/webhook/set
```

Depois execute:

```powershell
python scripts/set_webhook.py
```

Cole a URL HTTPS do Ngrok quando o script pedir. O webhook sera configurado para:

```text
{url_ngrok}/webhook
```

## 5. Teste pratico pelo WhatsApp

Envie uma mensagem para o numero conectado na Evolution API:

```text
Corrija: I is a student.
```

Resultado esperado:

- O FastAPI recebe o evento `messages.upsert`.
- O bot consulta o RAG com a aula cadastrada.
- A IA responde corrigindo para `I am a student`.
- O aluno recebe texto e audio no WhatsApp.
