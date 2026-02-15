# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kiwi Voice is a real-time Russian-language voice assistant integrated with the OpenClaw AI backend. It captures microphone audio, transcribes speech via Faster Whisper, detects the wake word "киви", identifies speakers via pyannote embeddings, communicates with OpenClaw over WebSocket, and speaks responses through configurable TTS providers.

All documentation and user-facing strings are in Russian. Code identifiers and comments mix English and Russian.

## Running the Service

```bash
# Activate the virtual environment
source venv/Scripts/activate   # Windows/MSYS2
source venv/bin/activate       # Linux

# Install dependencies
pip install -r requirements.txt

# Run the main service
python -m kiwi
```

Smoke tests: `pytest tests/test_smoke.py`

## Configuration

**Precedence:** `config.yaml` → environment variables (`.env`) → hardcoded defaults

- `config.yaml` — primary config (WebSocket, STT, TTS, wake word, VAD, speaker priority, security)
- `.env` — secrets and provider overrides (see `.env.example` for available vars)
- Key env vars for TTS routing: `KIWI_TTS_PROVIDER` (qwen3 | piper | elevenlabs), `KIWI_QWEN_BACKEND` (runpod | local)

## Architecture

### Audio Pipeline

```
Microphone (24kHz) → Audio Callback (energy + Silero VAD) → Audio Queue
  → KiwiListener._record_loop() → Faster Whisper STT → Wake Word Detection ("киви")
  → Speaker ID (pyannote embedding) → Priority Check (OWNER > FRIEND > GUEST > BLOCKED)
  → Voice Security (Telegram approval for dangerous commands from non-OWNER)
  → OpenClaw WebSocket (ws://127.0.0.1:18789, Protocol v3: chat.send → delta/final events)
  → TTS Provider → Speaker Output (with barge-in detection)
  → Loop back to listening
```

### Package Structure

```
kiwi-voice/
  kiwi/                          # Python package
    __init__.py                  # PROJECT_ROOT + version
    __main__.py                  # python -m kiwi
    service.py                   # Main orchestrator (KiwiServiceOpenClaw) + main()
    config_loader.py             # YAML/env config loading, KiwiConfig dataclass
    state_machine.py             # DialogueState definitions
    text_processing.py           # TTS text cleaning/splitting helpers
    utils.py                     # kiwi_log() + crash protection
    event_bus.py                 # Pub/sub event system (EventBus, EventType)
    listener.py                  # Audio recording, Whisper STT, wake word, VAD
    speaker_id.py                # Speaker embedding extraction (pyannote)
    speaker_manager.py           # Voice priority hierarchy + hot cache
    voice_security.py            # Dangerous command detection + Telegram approval
    unified_vad.py               # Voice Activity Detection pipeline
    hardware_aec.py              # Acoustic Echo Cancellation
    openclaw_ws.py               # WebSocket client for OpenClaw Gateway v3
    openclaw_cli.py              # CLI client for OpenClaw
    task_announcer.py            # Long-task status announcer
    tts/                         # TTS subpackage
      __init__.py                # Re-exports from base
      base.py                    # TTSProvider protocol, cache mixin, constants
      elevenlabs.py              # ElevenLabs TTS client
      piper.py                   # Local Piper TTS (ONNX)
      qwen_local.py              # Local Qwen3-TTS (GPU/CPU)
      runpod.py                  # RunPod serverless TTS client
      streaming.py               # Streaming TTS manager
  scripts/                       # Standalone utilities
    noise_monitor.py
    send_voice.py
    telegram_voice.py
  runpod/                        # Standalone RunPod deployment
  tests/
    test_smoke.py
  sounds/                        # Audio assets
  models/                        # ML models
  piper-models/                  # Piper ONNX models
  voice_profiles/                # Speaker profiles
  tts_cache/                     # TTS disk cache
  config.yaml                    # Runtime configuration
  pyproject.toml                 # Package metadata
```

### Speaker Priority System

```python
class VoicePriority(IntEnum):
    SELF = -1      # TTS echo filtering
    OWNER = 0      # Full access, cannot be blocked
    FRIEND = 1     # Dangerous commands need Telegram approval
    GUEST = 2      # All potentially dangerous commands need approval
    BLOCKED = 99   # Blacklisted
```

## Code Patterns

### Logging

Use `kiwi_log()` from `kiwi.utils` — never bare `print()`:
```python
from kiwi.utils import kiwi_log
kiwi_log("TAG", "message", level="INFO")  # → [14:08:25.342] [INFO] [TAG] message
```

### Project Root Paths

Use `PROJECT_ROOT` for paths to project-level assets:
```python
from kiwi import PROJECT_ROOT
path = os.path.join(PROJECT_ROOT, 'sounds', 'startup.mp3')
```

### Optional Module Loading

Modules are imported with try/except and availability flags:
```python
try:
    from kiwi.speaker_manager import SpeakerManager
    SPEAKER_MANAGER_AVAILABLE = True
except ImportError:
    SPEAKER_MANAGER_AVAILABLE = False
```

### Threading

All background work uses daemon threads with crash protection (try/except + sleep + continue in loops). Shared resources (cache, stdout, WebSocket) are guarded by `threading.Lock`.

### GPU Auto-Detection

CUDA is used when available, with automatic CPU fallback:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Windows UTF-8

Console codepage is set for Cyrillic output via `ctypes.windll.kernel32.SetConsoleCP(65001)`.

## Key Documentation

- `AI_NOTES.md` — detailed architecture notes, roadmap phases, and changelog (in Russian)
- `SKILL.md` — voice commands, security hierarchy, and deployment info (in Russian)

## Roadmap Status (from AI_NOTES.md)

- Phase 1: Stability & Observability — **done**
- Phase 2: State machine (IDLE → LISTENING → PROCESSING → SPEAKING) — **done**
- Phase 3: Streaming TTS (sentence-aware splitting) — **done**
- Phase 4: WebSocket OpenClaw integration — **done**
- Phase 5: Package structure reorganization — **done**
- Phase 6-7: Unified VAD, AEC, full event-driven architecture — pending
