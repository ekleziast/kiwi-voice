# Deployment Guide

## Quick Start (Local)

```bash
git clone https://github.com/ekleziast/kiwi-voice.git
cd kiwi-voice
python -m venv venv
source venv/bin/activate        # Linux/macOS
source venv/Scripts/activate    # Windows/MSYS2
pip install -r requirements.txt
cp .env.example .env            # Edit with your settings
python -m kiwi
```

Dashboard: http://localhost:7789

## Docker

```bash
docker build -t kiwi-voice .
docker run -d \
  --name kiwi-voice \
  --device /dev/snd \
  -v ./config.yaml:/app/config.yaml \
  -v ./.env:/app/.env \
  -v ./data:/app/data \
  -p 7789:7789 \
  kiwi-voice
```

### Docker Compose

```yaml
version: "3.8"
services:
  kiwi-voice:
    build: .
    restart: unless-stopped
    devices:
      - /dev/snd:/dev/snd
    volumes:
      - ./config.yaml:/app/config.yaml
      - ./.env:/app/.env
      - ./data:/app/data
    ports:
      - "7789:7789"
    environment:
      - KIWI_LANGUAGE=en
      - KIWI_TTS_PROVIDER=kokoro
```

## systemd (Linux)

Create `/etc/systemd/system/kiwi-voice.service`:

```ini
[Unit]
Description=Kiwi Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=kiwi
WorkingDirectory=/opt/kiwi-voice
ExecStart=/opt/kiwi-voice/venv/bin/python -m kiwi
Restart=on-failure
RestartSec=5
Environment=KIWI_LANGUAGE=en

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable kiwi-voice
sudo systemctl start kiwi-voice
sudo journalctl -u kiwi-voice -f  # view logs
```

## Configuration

### config.yaml

Primary configuration file. See `config.yaml` in the repo for all options.

Key sections:
- `language` — UI and voice language (en, ru, es, de, fr, ...)
- `tts.provider` — TTS engine (kokoro, piper, qwen3, elevenlabs)
- `stt.model` — Whisper model size (large, medium, small, tiny)
- `stt.device` — cuda or cpu
- `wake_word.engine` — openwakeword (ML) or text (fuzzy matching)
- `api.port` — REST API port (default: 7789)

### .env

Secrets and provider overrides. See `.env.example` for all available variables.

Key variables:
- `KIWI_LANGUAGE` — Override language
- `KIWI_TTS_PROVIDER` — Override TTS provider
- `KIWI_QWEN_BACKEND` — runpod or local
- `ELEVENLABS_API_KEY` — ElevenLabs API key
- `KIWI_TELEGRAM_BOT_TOKEN` — Telegram bot for voice security approvals
- `KIWI_TELEGRAM_CHAT_ID` — Telegram chat ID for approvals

## GPU Support

CUDA is auto-detected. For GPU acceleration:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Apple Silicon (macOS)

MLX Whisper is auto-detected on Apple Silicon:

```bash
pip install lightning-whisper-mlx
```

Set in config.yaml:
```yaml
stt:
  engine: mlx-whisper
```

## Reverse Proxy (nginx)

```nginx
server {
    listen 443 ssl;
    server_name kiwi.example.com;

    ssl_certificate /etc/letsencrypt/live/kiwi.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/kiwi.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:7789;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Home Assistant

See `custom_components/kiwi_voice/` for the custom component.

1. Copy `custom_components/kiwi_voice/` to your HA `custom_components/` directory
2. Restart Home Assistant
3. Add integration via UI: Settings → Integrations → Add → Kiwi Voice
4. Enter Kiwi Voice API URL (e.g., `http://192.168.1.100:7789`)

## Health Check

```bash
curl http://localhost:7789/api/status
```

Expected: `{"state": "LISTENING", "is_running": true, ...}`
