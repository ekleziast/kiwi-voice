---
hide:
  - navigation
  - toc
---

<section class="kv-hero">
  <div class="kv-hero-badge">Open Source Voice Interface</div>
  <h1 class="kv-hero-title">Kiwi Voice</h1>
  <p class="kv-hero-sub">
    ML wake word detection, speaker identification, voice&#8209;gated security,
    5&nbsp;TTS engines, 15&nbsp;languages, and a real&#8209;time web&nbsp;dashboard —
    for&nbsp;your own AI&nbsp;stack.
  </p>
  <div class="kv-hero-actions">
    <a href="getting-started/installation/" class="md-button md-button--primary">Get Started</a>
    <a href="https://github.com/ekleziast/kiwi-voice" class="md-button">GitHub</a>
  </div>
</section>

<section class="kv-section" markdown>

## How it works

Kiwi Voice turns your [OpenClaw](https://github.com/openclaw/openclaw) agent into a hands-free assistant. It captures audio from your microphone (or directly from the browser), detects the wake word, transcribes speech locally, identifies *who* is speaking, enforces security policies, sends the command to any LLM through OpenClaw's WebSocket gateway, and speaks the response back — all in a continuous loop.

```
You:  "Kiwi, turn on the lights in the bedroom"

Kiwi: [identifies speaker as Owner → full access]
      [sends to OpenClaw → routes to Home Assistant]
      "Done, the bedroom lights are on."
```

Think Alexa or Siri, but self-hosted, privacy-first, and plugged into your own AI stack.

</section>

<section class="kv-section">
  <h2>Features</h2>
  <div class="kv-grid">

    <a class="kv-card" href="features/wake-word/">
      <span class="kv-card-icon">🗣️</span>
      <strong>Wake Word Detection</strong>
      <span>Text fuzzy matching or <b>OpenWakeWord ML</b> — ONNX model, ~80ms latency, ~2% CPU. Built-in models or train your own.</span>
    </a>

    <a class="kv-card" href="features/speaker-id/">
      <span class="kv-card-icon">🎭</span>
      <strong>Speaker Identification</strong>
      <span>Voiceprint recognition via pyannote embeddings. Priority hierarchy: Owner → Friend → Guest → Blocked.</span>
    </a>

    <a class="kv-card" href="features/voice-security/">
      <span class="kv-card-icon">🔐</span>
      <strong>Two-Layer Security</strong>
      <span>Pre-LLM dangerous command detector + post-LLM exec approval. Telegram notifications for non-owner actions.</span>
    </a>

    <a class="kv-card" href="features/tts-providers/">
      <span class="kv-card-icon">🔊</span>
      <strong>5 TTS Providers</strong>
      <span>ElevenLabs, Kokoro ONNX, Piper, Qwen3-TTS. Streaming sentence-aware chunking with barge-in support.</span>
    </a>

    <a class="kv-card" href="features/web-dashboard/">
      <span class="kv-card-icon">📊</span>
      <strong>Web Dashboard & API</strong>
      <span>Glassmorphism dark dashboard with live status, event log, personalities, speaker management, and browser mic.</span>
    </a>

    <a class="kv-card" href="features/home-assistant/">
      <span class="kv-card-icon">🏠</span>
      <strong>Home Assistant</strong>
      <span>Bidirectional integration. Control Kiwi from HA dashboard, control your smart home by voice through Kiwi.</span>
    </a>

    <a class="kv-card" href="features/multilanguage/">
      <span class="kv-card-icon">🌍</span>
      <strong>15 Languages</strong>
      <span>Full i18n with YAML locales. All strings, voice commands, wake word variants, and security patterns per-language.</span>
    </a>

    <a class="kv-card" href="features/souls/">
      <span class="kv-card-icon">🎭</span>
      <strong>Personality System</strong>
      <span>5 built-in "souls" — switch by voice, API, or dashboard. NSFW routes to a separate isolated LLM session.</span>
    </a>

  </div>
</section>

<section class="kv-section" markdown>

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

</section>

<section class="kv-section" markdown>

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

</section>
