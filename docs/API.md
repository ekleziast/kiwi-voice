# Kiwi Voice REST API

Base URL: `http://localhost:7789`

## Status & Config

### `GET /api/status`

Returns current service state and metrics.

```json
{
  "state": "LISTENING",
  "language": "en",
  "tts_provider": "kokoro",
  "is_speaking": false,
  "is_processing": false,
  "is_running": true,
  "uptime_seconds": 3600,
  "active_speaker": "Owner",
  "active_soul": "default",
  "homeassistant_connected": true
}
```

### `GET /api/config`

Returns current configuration (safe fields only, no secrets).

```json
{
  "language": "en",
  "tts_provider": "kokoro",
  "tts_qwen_backend": "local",
  "tts_voice": "af_heart",
  "stt_model": "large",
  "stt_device": "cuda",
  "wake_word": "kiwi",
  "wake_word_engine": "openwakeword"
}
```

### `PATCH /api/config`

Update configuration at runtime.

**Body:**
```json
{
  "language": "ru",
  "wake_word": "jarvis",
  "tts_default_style": "cheerful"
}
```

**Response:** `{"updated": {"language": "ru"}}`

## Speakers

### `GET /api/speakers`

List all known speaker profiles.

```json
{
  "speakers": [
    {
      "id": "spk_001",
      "name": "Owner",
      "priority": 0,
      "is_blocked": false,
      "auto_learned": false,
      "sample_count": 42,
      "last_seen": "2026-02-25T10:30:00"
    }
  ]
}
```

### `DELETE /api/speakers/{speaker_id}`

Remove a speaker profile. **Response:** `{"deleted": "spk_001"}`

### `POST /api/speakers/{speaker_id}/block`

Block a speaker. **Response:** `{"blocked": "spk_001"}`

### `POST /api/speakers/{speaker_id}/unblock`

Unblock a speaker. **Response:** `{"unblocked": "spk_001"}`

## Languages

### `GET /api/languages`

```json
{
  "current": "en",
  "available": ["en", "ru", "es", "pt", "fr", "it", "de", "tr", "pl", "zh", "ja", "ko", "hi", "ar", "id"]
}
```

### `POST /api/language`

Switch language at runtime.

**Body:** `{"language": "ru"}`
**Response:** `{"language": "ru"}`

## Souls (Personalities)

### `GET /api/souls`

List all available personalities.

```json
{
  "souls": [
    {"id": "default", "name": "Default", "description": "Balanced assistant", "nsfw": false},
    {"id": "comedian", "name": "Comedian", "description": "Funny and witty", "nsfw": false},
    {"id": "siren", "name": "Siren", "description": "Flirty 18+", "nsfw": true}
  ]
}
```

### `GET /api/soul/current`

```json
{"id": "default", "name": "Default", "description": "...", "nsfw": false, "model": "claude-sonnet"}
```

### `POST /api/soul`

Switch personality.

**Body:** `{"soul": "comedian"}`
**Response:** `{"soul": "comedian", "name": "Comedian", "nsfw": false, "model": "claude-sonnet"}`

## TTS

### `POST /api/tts/test`

Speak a test phrase.

**Body:** `{"text": "Hello, I am Kiwi!"}` (optional, uses default if omitted)
**Response:** `{"status": "speaking", "text": "Hello, I am Kiwi!"}`

## Controls

### `POST /api/stop`

Stop current TTS playback. **Response:** `{"status": "stopped"}`

### `POST /api/reset-context`

Reset conversation context. **Response:** `{"status": "context_reset"}`

### `POST /api/restart`

Restart the service. **Response:** `{"status": "restarting"}`

### `POST /api/shutdown`

Shutdown the service. **Response:** `{"status": "shutting_down"}`

## Home Assistant

### `GET /api/homeassistant/status`

```json
{"enabled": true, "connected": true}
```

### `POST /api/homeassistant/command`

Send a voice command to Home Assistant.

**Body:** `{"text": "turn on bedroom lights", "language": "en"}`
**Response:** `{"response": "Done, bedroom lights are on.", "command": "turn on bedroom lights"}`

## WebSocket Events

### `GET /api/events` (WebSocket)

Real-time event stream. Connect via WebSocket to receive all EventBus events.

**Event format:**
```json
{
  "event": "WAKE_WORD_DETECTED",
  "data": {"text": "kiwi"},
  "timestamp": 1709000000.0,
  "source": "listener"
}
```

**Client commands:**
- `{"type": "ping"}` → `{"event": "pong"}`

**Event types:** `STATE_CHANGED`, `WAKE_WORD_DETECTED`, `SPEECH_RECOGNIZED`, `SPEAKER_IDENTIFIED`, `TTS_STARTED`, `TTS_FINISHED`, `LLM_TOKEN`, `LLM_COMPLETE`, `APPROVAL_REQUESTED`, `APPROVAL_RESOLVED`, `EXEC_APPROVAL_REQUESTED`, `EXEC_APPROVAL_RESOLVED`, `SOUL_CHANGED`, `ERROR`
