#!/usr/bin/env python3
"""Task status announcer for Kiwi Voice."""

import threading
import time
from typing import Callable, Optional

from kiwi.utils import kiwi_log


class TaskStatusAnnouncer:
    """Озвучивает статус длительных задач на основе контекста от LLM.

    Отслеживает стриминговый текст от LLM и периодически озвучивает
    краткие статусы, чтобы пользователь понимал что происходит.
    """

    def __init__(self, speak_func: Callable, intervals: list = None):
        """
        Args:
            speak_func: Функция для озвучивания (_speak_chunk)
            intervals: Интервалы оповещений в секундах [10, 30, 60, 120, 180]
        """
        self.speak_func = speak_func
        self.intervals = intervals or [6, 20, 45, 90, 150]

        self._start_time = 0.0
        self._last_text = ""
        self._last_announce_time = 0.0
        self._announced_intervals = set()  # Какие интервалы уже озвучены
        self._pending_activity = ""
        self._last_activity_key = ""
        self._last_activity_time = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tts_is_playing = False
        self._lock = threading.Lock()

        # Минимальный интервал между оповещениями (защита от спама)
        self._min_interval = 8.0
        self._activity_min_interval = 4.0

    def start(self, command: str):
        """Запускает мониторинг задачи."""
        with self._lock:
            self._start_time = time.time()
            self._last_text = ""
            self._last_announce_time = 0.0
            self._announced_intervals.clear()
            self._pending_activity = ""
            self._last_activity_key = ""
            self._last_activity_time = 0.0
            self._stop_event.clear()
            self._tts_is_playing = False

        # Запускаем фоновый поток мониторинга
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        kiwi_log("STATUS-ANNOUNCER", f"Started monitoring for: {command[:50]}...", level="INFO")

    def on_text_update(self, accumulated_text: str):
        """Вызывается при каждом delta от LLM."""
        with self._lock:
            self._last_text = accumulated_text

    def on_tts_playing(self, is_playing: bool):
        """Основной TTS начал/закончил воспроизведение."""
        acquired = self._lock.acquire(timeout=1.0)
        if acquired:
            try:
                self._tts_is_playing = is_playing
            finally:
                self._lock.release()
        else:
            # Lock stuck (e.g. deadlock in monitor thread) — update without lock
            self._tts_is_playing = is_playing

    def on_activity(self, message: str):
        """Получает и по возможности быстро озвучивает шаг выполнения."""
        cleaned = (message or "").strip()
        if not cleaned:
            return

        with self._lock:
            key = cleaned.lower()
            if key == self._last_activity_key:
                return
            self._last_activity_key = key
            self._pending_activity = cleaned

        self._announce_pending_activity()

    def stop(self):
        """Останавливает мониторинг."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        kiwi_log("STATUS-ANNOUNCER", "Stopped", level="INFO")

    def _monitor_loop(self):
        """Фоновый поток мониторинга."""
        while not self._stop_event.is_set():
            try:
                time.sleep(1.0)  # Проверяем каждую секунду
                self._announce_pending_activity()

                elapsed = time.time() - self._start_time

                # Проверяем, нужно ли озвучить статус
                for interval in self.intervals:
                    if elapsed >= interval and interval not in self._announced_intervals:
                        # Проверяем минимальный интервал с последнего оповещения
                        time_since_last = time.time() - self._last_announce_time
                        if time_since_last < self._min_interval:
                            continue

                        # Не озвучиваем если основной TTS играет
                        with self._lock:
                            if self._tts_is_playing:
                                continue

                            # Формируем статусное сообщение
                            status_msg = self._generate_status_message(elapsed)

                        if status_msg:
                            kiwi_log("STATUS-ANNOUNCER", f"Announcing at {elapsed:.0f}s: {status_msg[:50]}...", level="INFO")
                            self.speak_func(status_msg)

                            with self._lock:
                                self._announced_intervals.add(interval)
                                self._last_announce_time = time.time()

                        break  # Озвучили один интервал, ждём следующего

            except Exception as e:
                kiwi_log("STATUS-ANNOUNCER", f"Error in monitor loop: {e}", level="ERROR")

    def _announce_pending_activity(self):
        """Озвучивает отложенную активность, если сейчас это уместно."""
        msg_to_speak = ""
        now = time.time()
        with self._lock:
            if not self._pending_activity:
                return
            if self._tts_is_playing:
                return
            if (now - self._last_activity_time) < self._activity_min_interval:
                return
            msg_to_speak = self._pending_activity
            self._pending_activity = ""
            self._last_activity_time = now
            self._last_announce_time = now

        kiwi_log("STATUS-ANNOUNCER", f"Activity: {msg_to_speak[:60]}...", level="INFO")
        self.speak_func(msg_to_speak)

    def _generate_status_message(self, elapsed: float) -> str:
        """Генерирует статусное сообщение на основе контекста.

        IMPORTANT: caller must hold self._lock already.
        """
        text = self._last_text

        # Если текста нет — используем общие фразы
        if not text or len(text) < 10:
            if elapsed < 15:
                return "Думаю над ответом..."
            elif elapsed < 40:
                return "Задача сложная, работаю..."
            elif elapsed < 90:
                return "Всё ещё работаю над этим..."
            else:
                return "Продолжаю работу..."

        # Если есть текст — извлекаем последнее предложение как контекст
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            last_sentence = sentences[-1]

            # Обрезаем если слишком длинное
            if len(last_sentence) > 80:
                # Берём первые 60 символов + "..."
                last_sentence = last_sentence[:60].rsplit(' ', 1)[0] + "..."

            # Если предложение описывает действие — озвучиваем его
            action_keywords = [
                'анализирую', 'проверяю', 'ищу', 'создаю', 'исправляю',
                'читаю', 'пишу', 'обрабатываю', 'загружаю', 'сохраняю',
                'компилирую', 'тестирую', 'запускаю', 'останавливаю',
                'устанавливаю', 'удаляю', 'обновляю', 'настраиваю'
            ]

            if any(keyword in last_sentence.lower() for keyword in action_keywords):
                return last_sentence

        # Fallback: общая фраза с намёком на прогресс
        if elapsed < 40:
            return "Работаю над задачей..."
        elif elapsed < 90:
            return "Всё ещё работаю..."
        else:
            return "Продолжаю..."
