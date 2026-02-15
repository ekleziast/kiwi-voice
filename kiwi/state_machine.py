#!/usr/bin/env python3
"""Dialogue state definitions for Kiwi Voice."""


class DialogueState:
    """Состояния диалога - state machine для управления процессом."""
    IDLE = "idle"           # Ожидание wake word
    LISTENING = "listening" # Слушаем команду
    PROCESSING = "processing" # Проверяем completeness/intent (LLM busy)
    THINKING = "thinking"   # Ждём ответа от OpenClaw
    SPEAKING = "speaking"   # TTS воспроизведение
