---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Kiwi Voice

**Open-source voice interface for AI agents** — ML wake word detection, speaker identification,
voice-gated security, 5 TTS engines, 15 languages, and a real-time web dashboard.

[Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/ekleziast/kiwi-voice){ .md-button }

</div>

---

## How it works

Kiwi Voice turns your [OpenClaw](https://github.com/openclaw/openclaw) agent into a hands-free assistant. It captures audio from your microphone (or directly from the browser), detects the wake word, transcribes speech locally, identifies *who* is speaking, enforces security policies based on voice, sends the command to any LLM through OpenClaw's WebSocket gateway, and speaks the response back — all in a continuous loop.

```
You: "Kiwi, turn on the lights in the bedroom"

Kiwi: [identifies speaker as Owner → full access]
      [sends to OpenClaw → routes to Home Assistant]
      "Done, the bedroom lights are on."
```

Think Alexa or Siri, but self-hosted, privacy-first, and plugged into your own AI stack.

---

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### :material-microphone: Wake Word Detection

Text-based fuzzy matching or **OpenWakeWord ML** — a small ONNX model that listens to raw audio with ~80ms latency and ~2% CPU. Built-in models: `hey_jarvis`, `alexa`, `hey_mycroft`. [Train your own →](features/wake-word.md)

</div>

<div class="feature-card" markdown>

### :material-account-voice: Speaker Identification

Voiceprint recognition via pyannote embeddings. Kiwi knows *who* is talking and enforces a priority hierarchy: Owner → Friend → Guest → Blocked. [Learn more →](features/speaker-id.md)

</div>

<div class="feature-card" markdown>

### :material-shield-lock: Two-Layer Security

**Pre-LLM:** regex-based dangerous command detector across 15 languages + Telegram approval. **Post-LLM:** exec approval when the agent tries to run shell commands. [Details →](features/voice-security.md)

</div>

<div class="feature-card" markdown>

### :material-volume-high: 5 TTS Providers

ElevenLabs, Kokoro ONNX, Piper, Qwen3-TTS (local GPU or RunPod). Streaming sentence-aware chunking — starts speaking before the LLM finishes. [Compare →](features/tts-providers.md)

</div>

<div class="feature-card" markdown>

### :material-web: Web Dashboard & API

Real-time glassmorphism dashboard with live status, event log, personality carousel, speaker management, and browser microphone. 18 REST endpoints + WebSocket events. [Explore →](features/web-dashboard.md)

</div>

<div class="feature-card" markdown>

### :material-home-automation: Home Assistant

Bidirectional integration. Control Kiwi from HA dashboard, control your smart home by voice through Kiwi via the Conversation API. [Setup →](features/home-assistant.md)

</div>

<div class="feature-card" markdown>

### :material-translate: 15 Languages

Full i18n with YAML locale files. All user-facing strings, voice commands, wake word variants, hallucination filters, and security patterns are per-language. [Languages →](features/multilanguage.md)

</div>

<div class="feature-card" markdown>

### :material-drama-masks: Personality System

5 built-in "souls" — Mindful Companion, Storyteller, Comedian, Hype Person, Siren (18+). Switch by voice, API, or dashboard. NSFW routes to a separate LLM session. [Souls →](features/souls.md)

</div>

</div>

---

## Quick Start

```bash
git clone https://github.com/ekleziast/kiwi-voice.git
cd kiwi-voice
pip install -r requirements.txt
cp .env.example .env
python -m kiwi
```

Open [http://localhost:7789](http://localhost:7789) for the web dashboard.

[Full installation guide →](getting-started/installation.md)

---

## Architecture

```
Mic (24kHz) / Browser WebSocket → Audio Pipeline (Silero VAD + energy detection)
  → Wake Word (OpenWakeWord ML or text fuzzy match)
  → Faster Whisper STT (or MLX Whisper on Apple Silicon)
  → Speaker ID (pyannote embeddings) → Priority Gate (Owner/Friend/Guest/Blocked)
  → Voice Security (dangerous command regex → Telegram approval)
  → OpenClaw Gateway (WebSocket v3)
  → LLM response stream (delta → sentence chunking)
  → Streaming TTS (Kokoro/Piper/Qwen3/ElevenLabs) → Speaker output + browser playback
  → Barge-in detection → back to listening
```

[Architecture deep dive →](development/architecture.md)
