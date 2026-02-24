<p align="center">
  <img src="https://em-content.zobj.net/source/apple/391/kiwi-fruit_1f95d.png" width="120" alt="Kiwi Voice">
</p>

<h1 align="center">Kiwi Voice</h1>

<p align="center">
  <strong>OpenClaw voice assistant — speaker ID, voice-gated command approval, barge-in interrupts, and sentence-aware streaming TTS</strong>
</p>

<p align="center">
  <a href="https://github.com/ekleziast/kiwi-voice/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/openclaw/openclaw"><img src="https://img.shields.io/badge/backend-OpenClaw-orange.svg" alt="OpenClaw"></a>
</p>

<p align="center">
  <a href="README.ru.md">🇷🇺 Документация на русском</a>
</p>

---

## What is Kiwi Voice?

Kiwi Voice is a real-time voice interface that turns [OpenClaw](https://github.com/openclaw/openclaw) into a hands-free assistant. It captures audio from your microphone, recognizes speech locally via Faster Whisper, identifies *who* is speaking, enforces voice-based security policies, talks to any LLM through OpenClaw's WebSocket gateway, and speaks the response back — all in a continuous loop.

Think of it as Alexa/Siri, but self-hosted, privacy-first, and plugged into your own AI stack.

### Key Features

| Feature | Description |
|---------|-------------|
| 🗣️ **Wake Word** | Activate with a configurable keyword (default: *"kiwi"*) |
| 🎭 **Speaker ID** | Voiceprint recognition via pyannote embeddings — knows who's talking |
| 🔐 **Voice Security** | Priority hierarchy (Owner → Friend → Guest → Blocked) with Telegram approval for dangerous commands |
| 🔊 **Multi-Provider TTS** | ElevenLabs (cloud), Piper (local/free), Qwen3-TTS (local GPU / RunPod serverless) |
| ⚡ **Streaming TTS** | Sentence-aware chunking — starts speaking before the LLM finishes |
| 🛑 **Barge-In** | Interrupt the assistant mid-sentence by speaking over it |
| 🧠 **Auto-Learning** | Automatically remembers new voices after first interaction |
| 🔌 **WebSocket** | Native OpenClaw Gateway v3 protocol with delta/final streaming |
| 🌍 **Multi-Language** | Built-in i18n with YAML locale files — switch language with a single config field |

## Architecture

```
Mic → VAD + Energy Detection → Faster Whisper STT → Wake Word Check
  → Speaker ID (pyannote) → Priority Gate → Voice Security
  → OpenClaw Gateway (WebSocket) → LLM response stream
  → Real-time streaming TTS → Speaker Output (with barge-in)
  → Back to listening
```

## Quick Start

### Requirements

- **Python 3.10+**
- **FFmpeg** (for audio processing)
- **[OpenClaw](https://github.com/openclaw/openclaw)** running locally
- **GPU with CUDA** recommended (for STT & local TTS), but not required

### Installation

```bash
git clone https://github.com/ekleziast/kiwi-voice.git
cd kiwi-voice

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Fill in your API keys (ElevenLabs, RunPod, Telegram — all optional)
```

Edit `config.yaml` to match your setup:

```yaml
# Language: controls UI strings, STT, TTS, wake word, and command patterns
language: "ru"               # ru | en (add more in kiwi/locales/)

# TTS provider: elevenlabs | piper | qwen3
tts:
  provider: "piper"          # Free, local, no API key needed

# STT model
stt:
  model: "small"             # small = fast, large = accurate
  device: "cuda"             # cuda | cpu

# Wake word
wake_word:
  keyword: "kiwi"

# Owner name (used for voice commands like "I'm <name>")
speaker_priority:
  owner:
    name: "Owner"            # Change to your name
```

### Run

```bash
python -m kiwi
```

Or use the launcher scripts:

```bash
# Windows
start.bat
.\start.ps1

# Linux / macOS
python -m kiwi
```

## TTS Providers

| Provider | Quality | Latency | Cost | Local GPU |
|----------|---------|---------|------|-----------|
| **ElevenLabs** | Excellent | ~0.3–0.5s | ~$0.30/1K chars | No |
| **Qwen3-TTS (local)** | High | ~1–3s | Free | Yes (CUDA) |
| **Qwen3-TTS (RunPod)** | High | ~2–5s | ~$0.0003/sec | No |
| **Piper** | Good | <0.5s | Free | No |

Switch providers in `config.yaml` or via environment variable:

```bash
KIWI_TTS_PROVIDER=piper python -m kiwi
```

## Voice Security

Kiwi identifies speakers by voiceprint and enforces a priority hierarchy:

```
OWNER (priority 0)   — Full access, cannot be blocked
FRIEND (priority 1)  — Dangerous commands require Telegram approval
GUEST (priority 2)   — All sensitive commands require approval
BLOCKED (priority 99) — Completely ignored
```

### Voice Commands

| Command | Action |
|---------|--------|
| *"Kiwi, remember my voice"* | Register your voiceprint as owner |
| *"Kiwi, this is my friend [name]"* | Add someone as a friend |
| *"Kiwi, block them"* | Block the last speaker |
| *"Kiwi, who is speaking?"* | Identify the current speaker |
| *"Kiwi, what voices do you know?"* | List all known voiceprints |

> 💡 Voice commands are language-dependent. Set `language` in `config.yaml` to match your locale. See `kiwi/locales/*.yaml` for the full command lists.

### Telegram Approval

When a non-owner speaker issues a potentially dangerous command, Kiwi sends a confirmation request to the owner via Telegram. The owner can approve or deny it from their phone.

Set `KIWI_TELEGRAM_BOT_TOKEN` and `KIWI_TELEGRAM_CHAT_ID` in `.env` to enable.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KIWI_ELEVENLABS_API_KEY` | ElevenLabs API key |
| `RUNPOD_API_KEY` | RunPod API key (for Qwen3-TTS serverless) |
| `RUNPOD_TTS_ENDPOINT_ID` | RunPod endpoint ID |
| `KIWI_TELEGRAM_BOT_TOKEN` | Telegram bot token (voice security) |
| `KIWI_TELEGRAM_CHAT_ID` | Telegram chat ID for approval messages |
| `KIWI_TTS_PROVIDER` | Override TTS provider |
| `KIWI_FFMPEG_PATH` | Custom FFmpeg path |
| `KIWI_LANGUAGE` | Override language/locale (`ru`, `en`, etc.) |
| `KIWI_DEBUG` | Enable debug logging |
| `LLM_MODEL` | Override LLM model |

See `.env.example` for the full list.

## Project Structure

```
kiwi-voice/
├── kiwi/                    # Main Python package
│   ├── service.py           # Core orchestrator
│   ├── listener.py          # Audio capture, Whisper STT, wake word, VAD
│   ├── speaker_id.py        # Voiceprint extraction (pyannote)
│   ├── speaker_manager.py   # Priority hierarchy + hot cache
│   ├── voice_security.py    # Dangerous command detection + Telegram approval
│   ├── openclaw_ws.py       # WebSocket client (OpenClaw Gateway v3)
│   ├── config_loader.py     # YAML/env config loading
│   ├── text_processing.py   # Text cleanup and sentence splitting for TTS
│   ├── unified_vad.py       # Voice Activity Detection
│   ├── hardware_aec.py      # Acoustic Echo Cancellation
│   ├── task_announcer.py    # Long-running task status announcer
│   ├── i18n.py              # Lightweight i18n module (YAML locale loader)
│   ├── locales/             # Locale files (ru.yaml, en.yaml, ...)
│   ├── mixins/              # Service mixin modules
│   │   ├── audio_playback.py    # Audio output and playback control
│   │   ├── dialogue_pipeline.py # LLM dialogue orchestration
│   │   ├── llm_callbacks.py     # LLM streaming callbacks
│   │   ├── stream_watchdog.py   # Stream stall detection and recovery
│   │   └── tts_speech.py        # TTS synthesis and streaming playback
│   └── tts/                 # TTS providers
│       ├── elevenlabs.py
│       ├── elevenlabs_ws.py  # WebSocket input streaming
│       ├── piper.py
│       ├── qwen_local.py
│       ├── runpod.py
│       └── streaming.py     # Sentence-aware streaming manager
├── runpod/                  # RunPod serverless deployment (Qwen3-TTS)
├── scripts/                 # Utility scripts
├── sounds/                  # Audio assets (startup, confirmation, idle)
├── tests/                   # Smoke tests
├── config.yaml              # Main configuration
├── .env.example             # Secret template
└── pyproject.toml           # Package metadata
```

## Development

```bash
# Run tests
pytest tests/

# Code conventions:
# - Logging: kiwi_log("TAG", "message", level="INFO") — never print()
# - Paths: PROJECT_ROOT from kiwi package
# - Optional modules: try/except + *_AVAILABLE flags
# - Threads: daemon threads + crash protection
# - GPU: auto-detect CUDA with CPU fallback
```

## Multi-Language Support

Kiwi uses YAML-based locale files in `kiwi/locales/`. All user-facing strings, voice commands, wake word variants, hallucination filters, and security patterns are loaded from locale files.

**Switch language:**
```yaml
# config.yaml
language: "en"   # or "ru", etc.
```

**Add a new language:**
1. Copy `kiwi/locales/en.yaml` to `kiwi/locales/{lang}.yaml`
2. Translate all strings
3. Set `language: "{lang}"` in `config.yaml`

Currently shipped — **15 languages:**

| | | | |
|---|---|---|---|
| `ru` Russian | `en` English | `es` Spanish | `pt` Portuguese |
| `fr` French | `it` Italian | `de` German | `tr` Turkish |
| `pl` Polish | `zh` Chinese | `ja` Japanese | `ko` Korean |
| `hi` Hindi | `ar` Arabic | `id` Indonesian | |

## Roadmap

- [ ] Web UI for configuration
- [ ] Plugin system for custom wake words
- [ ] Home Assistant integration

## License

[MIT](LICENSE) — do whatever you want with it.

---

<p align="center">
  Built with 🥝 and too much coffee
</p>
