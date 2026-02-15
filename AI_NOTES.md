# Kiwi Voice Service - Технические заметки и Roadmap

**Последнее обновление:** 2026-02-12

---

## 📋 Содержание

1. [Текущая архитектура](#текущая-архитектура)
2. [Bug Fixes (активные)](#bug-fixes-активные)
3. [Roadmap: Realtime Voice Assistant](#roadmap-realtime-voice-assistant)
4. [История изменений](#история-изменений)

---

## Текущая архитектура

### Пайплайн обработки команды

```
Микрофон → [Record Loop] → [Audio Queue] → [Process Loop] → [Transcribe] →
[Wake Word Detect] → [Quick Check] → [OpenClaw Chat] → TTS
```

### Основные компоненты

| Файл | Назначение | Статус |
|------|------------|--------|
| `kiwi_service_openclaw.py` | Главный сервис, state machine (неактивен), интеграция с OpenClaw | ✅ Работает |
| `listener.py` | Запись аудио, Whisper STT, wake word detection, VAD | ✅ Работает |
| `piper_tts.py` | Локальный TTS | ✅ Работает |
| `speaker_manager.py` | Приоритеты голосов, OWNER/FRIENDS/GUESTS | ✅ Работает |
| `voice_security.py` | Telegram approval для опасных команд | ✅ Работает |

---

## Bug Fixes (активные)

### ✅ FIX 1: Timestamp Logging (2026-02-12)

**Проблема:** Всё логи `print("[TAG] ...")` без времени — сложно отлаживать асинхронные события.

**Решение:** Утилита `kiwi_log()` → `[HH:MM:SS.mmm] [TAG] msg`

```python
# Было:
print(f"[MIC] Speech started: vol={volume:.4f}")

# Стало:
kiwi_log("MIC", f"Speech started: vol={volume:.4f}")
# → [14:08:25.342] [MIC] Speech started: vol=0.0210
```

**Файлы затронуты:**
- `utils.py` (новый) — функция `kiwi_log()`
- `listener.py` — замена всех print
- `kiwi_service_openclaw.py` — замена всех print

---

### ✅ FIX 2: Crash Protection (2026-02-12)

**Проблема:** Скрипт неожиданно закрывается — вероятно из-за непойманного исключения в daemon-потоках.

**Корневая причина:**
1. `sounddevice` callback (`audio_callback`) — если внутри него исключение, поток умирает молча
2. Все worker threads — `daemon=True`, при падении main они не поднимаются
3. Нет `sys.excepthook` для глобального перехвата

**Решение:**

```python
# utils.py — глобальная защита
def setup_crash_protection():
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        log_crash(exc_type, exc_value, exc_traceback)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    sys.excepthook = custom_excepthook

# Внутри каждого daemon-потока:
def _record_loop(self):
    while self.is_running:
        try:
            # ... основная логика ...
        except Exception as e:
            log("ERROR", f"Record loop crashed: {e}")
            time.sleep(1)  # Auto-retry с задержкой
            continue  # Пробуем снова
```

**Файлы затронуты:**
- `utils.py` (новый) — `setup_crash_protection()`, `log_crash()`, `kiwi_log()`
- `listener.py` — try/except в `_record_loop`, `_streaming_loop`, `_process_loop`
- `kiwi_service_openclaw.py` — try/except в `_on_wake_word`

---

### ✅ FIX 3: Remove Redundancies (2026-02-12)

#### 3.1 Dialog Timeout Deduplication

**Было:** Двойная проверка `_check_dialog_mode()` + `_dialog_timeout_loop()`

**Решение:** Оставить только `_dialog_timeout_loop()`. `_check_dialog_mode()` использует состояние из `_dialog_timeout_loop`.

#### 3.2 Duplicate Typo Fix

**Было:** `text.replace("киеве", "киви")` дублирует `WAKE_WORD_TYPOS["киеве"] = "киви"`

**Решение:** Убрать строковые replace, оставить только словарь `WAKE_WORD_TYPOS`.

#### 3.3 Extract is_kiwi_speaking

**Было:** Копипаста `hasattr(self.on_wake_word, '__self__')` 3 раза в `listener.py`

**Решение:** Добавить метод `_is_kiwi_speaking()` в `KiwiListener`.

#### 3.4 Dead Code: text_analyzer.py

**Было:** `_quick_completeness_check()` дублирует `text_analyzer.py:is_complete_sentence()` но `text_analyzer` не импортируется.

**Решение:** Либо импортировать, либо удалить `text_analyzer.py`.

#### 3.5 Unused State Machine

**Было:** `DialogueState` класс есть, методы `_set_state()`, `_get_state()` есть, но они **никогда не вызываются**.

**Решение:** Удалить мёртвый код или активировать. **Решено:** Удалить до момента реальной интеграции (см. Roadmap Phase 2).

#### 3.6 Double VAD Check

**Было:** VAD проверяется в `audio_callback` (начало записи) И в `_submit_audio` (перед отправкой).

**Решение:** Оставить только в `audio_callback` — там же VAD используется для extension logic (`_check_vad_continuation`).

**Файлы затронуты:**
- `listener.py` — рефакторинг `_submit_audio`, `_record_loop`, `_fix_transcription`
- `kiwi_service_openclaw.py` — удаление `DialogueState` и связанных методов
- `text_analyzer.py` — удалить (или оставить как reference)

---

## Roadmap: Realtime Voice Assistant

### Фаза 1: Stability & Observability ✅ (ТЕКУЩАЯ)
**Цель:** Скрипт не падает, логи информативны

- [x] Timestamp в логах
- [x] Crash protection (try/except во всех потоках)
- [x] Watchdog для daemon-тредов
- [x] Structured logging
- [x] Убрать избыточные проверки

**ETA:** 1 день

---

### Фаза 2: Activate State Machine 🔲
**Цель:** Подключить существующий `DialogueState` к реальным переходам

**Задачи:**
- [ ] Интегрировать `_set_state()` в `_on_wake_word()`
- [ ] State transitions: IDLE → LISTENING → PROCESSING → THINKING → SPEAKING
- [ ] State-based таймауты вместо ad-hoc проверок
- [ ] Логирование state transitions
- [ ] Визуализация текущего state в логах (`[STATE] Transition: IDLE → LISTENING`)

**Текущий код:**
```python
class DialogueState:
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    THINKING = "thinking"
    SPEAKING = "speaking"
```

**ETA:** 1-2 дня

---

### Фаза 3: Streaming TTS 🔲
**Цель:** Начать проигрывать TTS ДО того как сгенерирован весь ответ

**Текущая проблема:** Click-to-speech latency = Whisper (3s) + LLM (10s) + TTS (2s) = ~15s

**Решение:**
```
LLM отвечает: "Слышу. Работает, длинная запись прошла. Всё ок..."
                      ↓
Sentence 1: "Слышу." → TTS → play (0.5s)
Sentence 2: "Работает..." → TTS (параллельно)
Sentence 3: "Всё ок..." → TTS (параллельно)
                      ↓
Итог: Первая реакция через ~0.5s, не ~15s!
```

**Задачи:**
- [ ] Sentence-aware LLM wrapper (разбивает ответ на предложения)
- [ ] Sentence queue
- [ ] Parallel pipeline: `sentence → TTS → play`
- [ ] Buffering для smooth playback (убрать паузы между предложениями)

**ETA:** 3-5 дней

---

### Фаза 4: WebSocket OpenClaw 🔄 (В РАБОТЕ)
**Цель:** Streaming ответ от LLM (убрать subprocess overhead)

**Текущая проблема:** `subprocess.run()` — каждый вызов запускает Node.js (~1s overhead)

**Архитектура WebSocket:**
```
Kiwi ←→ WebSocket ←→ OpenClaw Gateway (ws://127.0.0.1:18789)
```

**Протокол Gateway v3 (исправлено):**

**Константы:**
- `PROTOCOL_VERSION = 3`
- `WS_URL = ws://127.0.0.1:18789` (БЕЗ пути!)
- Gateway token из `~/.openclaw/openclaw.json` → `gateway.auth.token`

**Формат фреймов:**
```python
# REQUEST (клиент → сервер):
{"type": "req", "id": "<uuid4>", "method": "<method>", "params": {...}}

# RESPONSE (сервер → клиент):
{"type": "res", "id": "<same_uuid>", "ok": True/False, "payload": {...}}

# EVENT (сервер → клиент):
{"type": "event", "event": "<name>", "payload": {...}, "seq": int}
```

**Handshake (исправленный):**
```python
# 1. Получаем challenge:
{"type": "event", "event": "connect.challenge", "payload": {"nonce": "...", "ts": ...}}

# 2. Отправляем connect (строгая схема!):
{
    "type": "req",
    "id": str(uuid4()),
    "method": "connect",
    "params": {
        "minProtocol": 3,
        "maxProtocol": 3,
        "client": {
            "id": "gateway-client",  # из GATEWAY_CLIENT_IDS
            "version": "1.0.0",
            "platform": "win32",
            "mode": "backend"          # из GATEWAY_CLIENT_MODES
        },
        "role": "operator",
        "scopes": ["operator.admin"],
        "caps": [],
        "auth": {"token": "<gateway_token>"},
        "locale": "ru-RU",
        "userAgent": "kiwi-voice/1.0"
    }
}

# 3. Получаем hello-ok:
{"type": "res", "id": "...", "ok": true, "payload": {"type": "hello-ok", "protocol": 3, ...}}
```

**Отправка сообщения (chat.send):**
```python
{
    "type": "req",
    "id": str(uuid4()),
    "method": "chat.send",
    "params": {
        "sessionKey": "agent:main:main",
        "message": "Привет!",
        "idempotencyKey": str(uuid4()),
        "timeoutMs": 120000
    }
}
```

**Получение ответа (chat events):**
```python
# Стриминг:
{"type": "event", "event": "chat", "payload": {
    "runId": "...",
    "sessionKey": "agent:main:main",
    "seq": 0,
    "state": "delta",  # частичный ответ
    "message": {"content": "Прив"}
}}

# Финальный ответ:
{"type": "event", "event": "chat", "payload": {
    "seq": 5,
    "state": "final",  # полный ответ
    "message": {"content": "Привет! Как дела?"}
}}

# Ошибка:
{"type": "event", "event": "chat", "payload": {
    "state": "error",
    "errorMessage": "..."
}}
```

**Важные нюансы:**
- `additionalProperties: false` — схема строгая, нельзя добавлять лишние поля!
- `client.id` — enum из `GATEWAY_CLIENT_IDS`, использовать `"gateway-client"`
- `client.mode` — enum из `GATEWAY_CLIENT_MODES`, использовать `"backend"`
- Gateway token обязателен в `auth.token`
- URL без пути: `ws://127.0.0.1:18789` (не `/ws/agent/main`)
- `idempotencyKey` обязателен в `chat.send`
- `sessionKey` формат: `"agent:{agent_id}:{session_id}"` → `"agent:main:main"`

**Задачи:**
- [x] Разобрать протокол из исходников OpenClaw
- [x] Найти gateway token
- [x] Обновить AI_NOTES.md
- [x] Переписать `OpenClawWebSocket` с правильным handshake (connect.challenge → connect req → hello-ok)
- [x] Использовать `chat.send` вместо кастомных сообщений
- [x] Обработать `chat` events (delta/final/error/aborted)
- [x] `chat.abort` для отмены запросов
- [x] Автозагрузка gateway token из `~/.openclaw/openclaw.json`
- [x] URL без пути: `ws://127.0.0.1:18789`
- [ ] Тестирование (ручное)

**Статус:** ✅ Реализовано (2026-02-12), ожидает тестирования

**Изменённые файлы:**
- `kiwi_service_openclaw.py` — полностью переписан `OpenClawWebSocket`
- `config.yaml` — обновлены комментарии WebSocket секции, host → `127.0.0.1`
- `AI_NOTES.md` — обновлён статус

---

### Фаза 5: Unified VAD Pipeline 🔲
**Цель:** Event-driven turn detection вместо многоуровневого polling

**Текущая архитектура (проблемы):**
```
Energy Threshold → Silero VAD (audio_callback) → Whisper VAD (no_speech_prob) →
Fixed Silence Timer → Barge-in Polling (50ms)
```

**Решение — единый pipeline:**
```
Mic → Silero VAD (единственный) → Turn Detection Decision Engine → Events
```

**Turn Detection Decision Engine:**
- Input: VAD confidence stream
- Output: `speech_started`, `speech_continues`, `speech_ended` events
- Semantic turn end: LLM определяет "конец мысли" по partial STT

**Задачи:**
- [ ] Убрать energy threshold (только Silero VAD)
- [ ] Убрать Whisper VAD (проверка no_speech_prob в `_transcribe`)
- [ ] Sentence-aware turn detection
- [ ] Barge-in как event (не polling)

**ETA:** 2-3 дня

---

### Фаза 6: Hardware AEC (Acoustic Echo Cancellation) 🔲
**Цель:** Заменить speaker embedding echo cancellation на аппаратное AEC

**Текущая проблема:** Сравнение embeddings — медленно, ненадёжно

**Решения:**
1. **WebRTC AEC** (рекомендуется)
   - `py-webrtcvad` для VAD + `webrtc-aec` для эхоподавления
   - Требует loopback audio (знать что TTS проигрывается)
   
2. **SpeexDSP**
   - `speexdsp-python` — AEC + noise suppression
   - Проще в интеграции

3. **Аппаратный loopback**
   - Отдавать TTS аудио как reference в AEC библиотеку
   - Output: очищенный микрофонный сигнал

**Задачи:**
- [ ] Выбрать библиотеку (WebRTC vs SpeexDSP)
- [ ] Создать AEC wrapper
- [ ] Feed TTS audio as reference
- [ ] Убрать speaker embedding echo cancellation

**ETA:** 3-5 дней

---

### Фаза 7: Event-Based Architecture 🔲
**Цель:** Убрать оставшиеся polling loops

**Текущие polling loops:**
- `_barge_in_counter` — polling каждые 50ms
- `_dialog_timeout_loop` — polling каждые 500ms
- `while stream.active:` — polling TTS playback

**Решение — Event Bus:**
```python
class EventBus:
    def subscribe(event_type: str, callback: Callable)
    def publish(event: Event)

# Events:
# - "vad.speech_started"
# - "vad.speech_ended"
# - "tts.started"
# - "tts.completed"
# - "barge_in.requested"
# - "state.changed"
```

**Задачи:**
- [ ] Create `EventBus` class
- [ ] Convert VAD to event-driven
- [ ] Convert TTS playback to event-driven
- [ ] Convert barge-in to event-driven
- [ ] Convert state machine to event-driven

**ETA:** 2-3 дня

---

### Итоговая архитектура (после всех фаз)

```
┌─────────────────────────────────────────────────────────────┐
│                    Event Bus                                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  VAD    │   │  STT     │   │  LLM     │   │  TTS     │ │
│  │ Pipeline│──▶│ Streaming│──▶│ Streaming│──▶│ Streaming│ │
│  │         │   │ Whisper  │   │ WebSocket│   │ Sentence │ │
│  └────┬────┘   └──────────┘   └──────────┘   └────┬─────┘ │
│       │                                           │       │
│       │   speech_started ─────────► turn_detection│       │
│       │   speech_ended   ─────────►             │       │
│       │                                        ▼       │
│  ┌────┴─────────┐                          ┌──────────┐ │
│  │  AEC Module  │                          │  Speaker │ │
│  │ (reference)  │◄─────────────────────────│  Output  │ │
│  └──────────────┘                          └──────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Target Latency:**
- Wake word → First TTS byte: 500-800ms (вместо 15s)
- Full response: 2-5s (вместо 30-60s)

---

## История изменений

### ✅ FIX: Slow Model Loading + WebSocket Content Parsing (2026-02-12)

**Проблема 1:** STT модель загружается медленно (~30-45 секунд) из-за последовательной загрузки:
- pyannote/embedding (~10-15s) — загружалась в `SpeakerIdentifier.__init__`
- Silero VAD (~3-5s) — загружалась в `KiwiListener.__init__`
- Faster Whisper (~5-10s) — загружается в `listener.start()`
- WebSocket connect timeout (~15s если недоступен)

**Проблема 2:** WebSocket возвращает `content` как dict `{'type': 'text', 'text': '...'}`, а код ожидает строку. TTS читает вслух `"{'type': 'text', 'text': 'Привет!'}"` вместо `"Привет!"`.

**Проблема 3:** Логи WebSocket буферизуются и выводятся пачкой ("насрались").

**Решение:**
1. **Ленивая загрузка pyannote** — добавлен метод `_ensure_model_loaded()`, вызывается в `extract_embedding()` при первом использовании
2. **Ленивая загрузка Silero VAD** — добавлен метод `_ensure_vad_loaded()`, вызывается в `_check_vad()` при первом использовании
3. **Фикс парсинга content** — добавлена проверка `isinstance(content, dict)` в `_handle_chat_event()` с извлечением `content.get('text')`
4. **Flush логов** — добавлен `flush=True` в `_log_ws()` для немедленного вывода

**Результат:** 
- Время старта сократилось с ~30-45s до ~5-10s (только Whisper + калибровка шума)
- pyannote загрузится при первом создании self-profile (после первого TTS)
- Silero VAD загрузится при первой проверке barge-in
- TTS теперь корректно читает текст, а не JSON
- Логи WebSocket выводятся моментально

**Файлы:**
- `speaker_id.py` — ленивая загрузка pyannote (`_ensure_model_loaded()`)
- `listener.py` — ленивая загрузка Silero VAD (`_ensure_vad_loaded()`)
- `kiwi_service_openclaw.py` — фикс content parsing + flush логов
- `AI_NOTES.md` — обновлена документация

---

### ✅ FIX: CLI Parsing (2026-02-12)

**Проблема:** OpenClaw CLI возвращал пустой ответ из-за race condition в стриминговом чтении stdout.

**Решение:** `subprocess.run()` вместо `Popen` + `readline()`.

**Файлы:** `kiwi_service_openclaw.py`

---

### ✅ FIX: Disabled LLM Filter (2026-02-12)

Удалены дорогие LLM-вызовы (5-15s overhead). Теперь только `_quick_completeness_check()` (~1ms).

---

### ✅ FIX: VAD Sensitivity & Noise Recalibration (2026-02-16)

**Проблема:** Kiwi перестаёт слышать речь или обрезает фразы на середине.

**Корневая причина:**
1. Noise floor калибруется один раз при старте. Если при старте был шум, `_silence_threshold` залипает навсегда (наблюдалось: thr=0.0853, а реальный фон 0.0003)
2. Речь пользователя (vol=0.02-0.04) ниже порога (0.0853) — энергетический VAD считает её тишиной
3. VAD continuation check (строка 1477) требовал `volume >= _silence_threshold`, поэтому Silero VAD не мог продлить запись при тихой речи
4. Energy gate (0.012) отбрасывал записи с тихим голосом (rms=0.0107)
5. `silence_duration_end=1.5s` обрезал фразы слишком рано при средней длине речи

**Решение:**
1. **Непрерывная рекалибровка** — каждые ~30s тишины пересчитывает noise floor из реальных ambient сэмплов
2. **VAD continuation fix** — условие заменено с `_silence_threshold` на `effective_min_speech_volume` (обычно ~0.006-0.015), позволяя Silero VAD продлять запись тихой речи
3. **Energy gate снижен** — 0.012 → 0.006
4. **Config tuning** — `noise_threshold_multiplier` 1.5→1.3, `min_silence_threshold` 0.008→0.005, `silence_duration_end` 1.5→1.8

**Файлы:**
- `config.yaml` — VAD параметры
- `kiwi/listener.py` — рекалибровка, VAD continuation, energy gate, VAD override, noisereduce import
- `kiwi/unified_vad.py` — `energy_min_threshold` 0.004→0.008

---

### ✅ FEATURE: Device Identity & Session Isolation (2026-02-16)

**Проблема:** Gateway не различает устройства; события от чужих сессий обрабатывались Kiwi.

**Решение:**
1. **Ed25519 device identity** — генерируется при первом запуске, сохраняется в `device-identity.json`. Подписывает connect request (v2 payload)
2. **Session key filtering** — `_handle_lifecycle_event` и `_handle_chat_event` игнорируют события от чужих sessionKey

**Файлы:**
- `kiwi/openclaw_ws.py` — device auth, session filtering
- `.gitignore` — `device-identity.json` (приватный ключ)
- `requirements.txt` — `cryptography>=41.0.0`

---

### ✅ REFACTOR: Command Pipeline & Audio Stability (2026-02-16)

**Изменения:**
1. **CommandContext dataclass** — состояние команды (speaker, approval, abort) передаётся через stages вместо локальных переменных в 300-строчном `_on_wake_word`
2. **Pipeline stages** — `_on_wake_word` разбит на 8 стадий (`_stage_init_and_dedup`, `_stage_resolve_speaker`, `_stage_check_approval`, `_stage_handle_special_commands`, `_stage_handle_stop_cancel`, `_stage_completeness_check`, `_stage_owner_approval_gate`, `_stage_dispatch_to_llm`)
3. **`_sd_play_lock`** — защита от concurrent `sd.play()` вызовов (race между status announcer и response TTS)
4. **`_speak_chunk` guard** — не запускает synthesis если response уже играет
5. **Streaming TTS stop fix** — упрощена логика final chunk в `StreamingTTSManager.stop()`
6. **Whisper `no_speech_threshold`** — 0.6→0.85 (меньше ложных отклонений тихой речи)
7. **Noise reduction** — добавлен `noisereduce` (spectral gating, `prop_decrease=0.4`) для очистки аудио перед Whisper

**Файлы:**
- `kiwi/service.py` — CommandContext, pipeline, sd_play_lock, speak_chunk guard
- `kiwi/tts/streaming.py` — final chunk fix
- `kiwi/listener.py` — noisereduce, no_speech_threshold
- `requirements.txt` — `noisereduce>=3.0.0`

---

*Дополнить при реализации Phases.*
