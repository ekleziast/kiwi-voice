# Kiwi Voice

Голосовой ИИ-ассистент на русском языке с интеграцией в бэкенд [OpenClaw](https://github.com/openclaw). Захватывает аудио с микрофона, распознаёт речь через Faster Whisper, реагирует на wake word «киви», идентифицирует говорящего по голосу, общается с LLM через WebSocket и озвучивает ответы с помощью нескольких TTS-провайдеров.

## Возможности

- **Wake word** — активация по слову «киви» в начале фразы
- **Speaker ID** — идентификация говорящего по голосовому отпечатку (pyannote)
- **Мульти-TTS** — Qwen3-TTS (local/RunPod), Piper (local), ElevenLabs (cloud)
- **Система безопасности** — иерархия голосов (OWNER > FRIEND > GUEST > BLOCKED), Telegram-подтверждение опасных команд
- **Streaming TTS** — sentence-aware разбивка для естественного звучания
- **WebSocket** — интеграция с OpenClaw Gateway v3 (delta/final streaming)
- **Barge-in** — прерывание ответа голосом
- **Auto-learning** — автоматическое запоминание новых голосов

## Системные требования

### Hardware
- Микрофон (USB или встроенный)
- Колонки / наушники
- GPU с CUDA (рекомендуется для STT и локального TTS, но не обязательно)

### Software
- Python 3.10+
- FFmpeg (для обработки аудио)
- OpenClaw Gateway (WebSocket-бэкенд)

## Быстрый старт

### 1. Клонирование и установка

```bash
git clone https://github.com/ekleziast/kiwi-voice.git
cd kiwi-voice

python -m venv venv
source venv/bin/activate       # Linux/macOS
source venv/Scripts/activate   # Windows/MSYS2

pip install -r requirements.txt
```

### 2. Настройка

```bash
# Скопировать пример конфигурации секретов
cp .env.example .env
# Заполнить API-ключи в .env

# Отредактировать config.yaml под своё окружение
# (TTS-провайдер, модель STT, параметры VAD и т.д.)
```

### 3. Запуск

```bash
python -m kiwi
```

Или через лаунчеры:
```bash
# Windows
start.bat
# PowerShell
.\start.ps1
```

## Конфигурация

Приоритет: `config.yaml` → переменные окружения (`.env`) → значения по умолчанию.

### config.yaml

Основные секции:
- `websocket` — подключение к OpenClaw Gateway
- `llm` — модель, таймауты, системный промт
- `tts` — провайдер (elevenlabs/piper/qwen3), настройки голоса
- `stt` — модель Whisper, устройство, язык
- `wake_word` — ключевое слово и позиция
- `audio` — sample rate, устройства ввода/вывода
- `realtime` — VAD, barge-in, кэш TTS
- `security` — Telegram approval, авто-обучение
- `speaker_priority` — имя и ID владельца
- `dangerous_commands` — паттерны опасных команд

### Переменные окружения (.env)

| Переменная | Описание |
|------------|----------|
| `RUNPOD_TTS_ENDPOINT_ID` | ID эндпоинта RunPod (provider=qwen3, backend=runpod) |
| `RUNPOD_API_KEY` | API-ключ RunPod |
| `KIWI_ELEVENLABS_API_KEY` | API-ключ ElevenLabs (provider=elevenlabs) |
| `KIWI_TELEGRAM_BOT_TOKEN` | Токен Telegram-бота (voice security) |
| `KIWI_TELEGRAM_CHAT_ID` | Chat ID для Telegram approval |
| `KIWI_TTS_PROVIDER` | Провайдер TTS: elevenlabs / piper / qwen3 |
| `KIWI_QWEN_BACKEND` | Бэкенд Qwen3: runpod / local |
| `KIWI_FFMPEG_PATH` | Путь к директории с ffmpeg |
| `KIWI_DEBUG` | Включить отладочный лог при старте |
| `LLM_MODEL` | Модель LLM (например, openai/gpt-5.2) |

Полный список — в `.env.example`.

## Архитектура

### Аудио-пайплайн

```
Микрофон (24kHz) → Audio Callback (energy + Silero VAD) → Audio Queue
  → KiwiListener._record_loop() → Faster Whisper STT → Wake Word Detection ("киви")
  → Speaker ID (pyannote embedding) → Priority Check (OWNER > FRIEND > GUEST > BLOCKED)
  → Voice Security (Telegram approval для опасных команд от не-OWNER)
  → OpenClaw WebSocket (ws://127.0.0.1:18789, Protocol v3: chat.send → delta/final)
  → TTS Provider → Speaker Output (с barge-in detection)
  → Цикл обратно к прослушиванию
```

### Структура пакета

```
kiwi-voice/
  kiwi/                          # Python-пакет
    __init__.py                  # PROJECT_ROOT + version
    __main__.py                  # python -m kiwi
    service.py                   # Главный оркестратор (KiwiServiceOpenClaw)
    config_loader.py             # Загрузка YAML/env, KiwiConfig dataclass
    state_machine.py             # Состояния диалога
    text_processing.py           # Очистка/разбивка текста для TTS
    utils.py                     # kiwi_log() + crash protection
    event_bus.py                 # Pub/sub система событий
    listener.py                  # Запись аудио, Whisper STT, wake word, VAD
    speaker_id.py                # Извлечение голосовых эмбеддингов (pyannote)
    speaker_manager.py           # Иерархия приоритетов + hot cache
    voice_security.py            # Детекция опасных команд + Telegram approval
    unified_vad.py               # Voice Activity Detection
    hardware_aec.py              # Acoustic Echo Cancellation
    openclaw_ws.py               # WebSocket-клиент для OpenClaw Gateway v3
    openclaw_cli.py              # CLI-клиент для OpenClaw
    task_announcer.py            # Анонсер длительных задач
    tts/                         # TTS-подпакет
      base.py                    # Протокол TTSProvider, кэш, константы
      elevenlabs.py              # ElevenLabs TTS
      piper.py                   # Локальный Piper TTS (ONNX)
      qwen_local.py              # Локальный Qwen3-TTS (GPU/CPU)
      runpod.py                  # RunPod serverless TTS
      streaming.py               # Streaming TTS manager
  scripts/                       # Утилиты
  runpod/                        # Деплой RunPod serverless
  tests/                         # Тесты
  sounds/                        # Звуковые ассеты
  config.yaml                    # Конфигурация
  pyproject.toml                 # Метаданные пакета
```

## TTS-провайдеры

| Провайдер | Качество | Латентность | Стоимость | GPU |
|-----------|----------|-------------|-----------|-----|
| **Qwen3-TTS (local)** | Высокое | ~1-3 сек | Бесплатно | Да (CUDA) |
| **Qwen3-TTS (RunPod)** | Высокое | ~2-5 сек | ~$0.0003/сек | Нет |
| **Piper** | Среднее | <0.5 сек | Бесплатно | Нет |
| **ElevenLabs** | Отличное | ~1-2 сек | ~$0.30/1K символов | Нет |

Переключение провайдера:
```yaml
# config.yaml
tts:
  provider: "elevenlabs"  # elevenlabs | piper | qwen3
  qwen_backend: "local"   # runpod | local (для qwen3)
```

## Система безопасности голоса

### Иерархия приоритетов

```
PRIORITY 0   OWNER    — полный доступ, не может быть заблокирован
PRIORITY 1   FRIEND   — опасные команды требуют Telegram-подтверждения
PRIORITY 2   GUEST    — все потенциально опасные команды требуют подтверждения
PRIORITY 99  BLOCKED  — полный игнор
```

### Голосовые команды управления

| Команда | Действие |
|---------|----------|
| «Киви, запомни меня как [имя]» | Регистрация голоса |
| «Киви, это мой друг [имя]» | Добавление в FRIENDS |
| «Киви, добавь в чёрный список» | Блокировка последнего говорящего |
| «Киви, разблокируй» | Удаление из чёрного списка |
| «Киви, кто это говорит?» | Идентификация голоса |
| «Киви, какие голоса?» | Список запомненных голосов |
| «Киви, забудь этот голос» | Удаление профиля |

### Telegram Approval

Опасные команды от не-OWNER отправляются владельцу в Telegram для подтверждения. Настраивается через `KIWI_TELEGRAM_BOT_TOKEN` и `KIWI_TELEGRAM_CHAT_ID`.

## Разработка

### Тесты

```bash
pytest tests/test_smoke.py
```

### Паттерны кода

- **Логирование:** `kiwi_log("TAG", "message", level="INFO")` — никогда `print()`
- **Пути:** `PROJECT_ROOT` из `kiwi` для доступа к ассетам
- **Опциональные модули:** try/except + `*_AVAILABLE` флаги
- **Потоки:** daemon threads + crash protection + `threading.Lock`
- **GPU:** автодетекция CUDA с fallback на CPU

## Лицензия

[MIT](LICENSE)
