#!/usr/bin/env python3
"""Streaming TTS manager for Kiwi Voice."""

import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
import sounddevice as sd

from kiwi.utils import kiwi_log


class StreamingTTSManager:
    """Менеджер для потокового TTS во время генерации LLM.

    Накапливает токены от LLM и отправляет готовые предложения в TTS
    параллельно с генерацией следующих предложений.
    """

    def __init__(
        self,
        tts_callback: Callable,
        min_chunk_chars: int = 40,
        max_chunk_chars: int = 150,
        tts_synthesize_callback: Optional[Callable[[str], Optional[Tuple[np.ndarray, int]]]] = None,
        playback_callback: Optional[Callable[[np.ndarray, int], None]] = None,
        synthesis_workers: int = 1,
        max_chunk_wait_s: float = 20.0,
    ):
        self.tts_callback = tts_callback
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.tts_synthesize_callback = tts_synthesize_callback
        self.playback_callback = playback_callback
        self.synthesis_workers = max(1, int(synthesis_workers))
        self.max_chunk_wait_s = max(3.0, float(max_chunk_wait_s))
        self._buffer = ""
        self._sent_text = ""
        self._lock = threading.Lock()
        self._playback_cond = threading.Condition(self._lock)
        self._is_active = False
        self._playback_thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[int, Any] = {}
        self._next_chunk_id = 0
        self._next_play_id = 0
        self._finalized = False
        self._graceful_shutdown = True
        self._stop_event = threading.Event()

    def start(self):
        """Запускает менеджер стриминга."""
        with self._lock:
            self._buffer = ""
            self._sent_text = ""
            self._is_active = True
            self._futures = {}
            self._next_chunk_id = 0
            self._next_play_id = 0
            self._finalized = False
            self._graceful_shutdown = True
            self._stop_event.clear()
            self._executor = ThreadPoolExecutor(
                max_workers=self.synthesis_workers,
                thread_name_prefix="kiwi-tts-synth",
            )

        # Отдельный поток воспроизводит чанки строго по порядку.
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()
        kiwi_log("STREAM-TTS", "Manager started", level="INFO")

    def stop(self, graceful: bool = True):
        """Останавливает менеджер стриминга.

        graceful=True: договаривает всё, что уже в очереди/буфере.
        graceful=False: немедленно прерывает текущую озвучку и очищает очередь.
        """
        with self._lock:
            self._is_active = False
            self._graceful_shutdown = graceful
            if graceful:
                # Отправляем остаток буфера если есть
                remaining = self._buffer.strip()
                if remaining and len(remaining) > len(self._sent_text):
                    final_chunk = remaining[len(self._sent_text):].strip()
                    if final_chunk:
                        kiwi_log(
                            "STREAM-TTS",
                            f"Queuing final chunk ({len(final_chunk)} chars): {final_chunk[:50]}...",
                            level="INFO",
                        )
                        self._enqueue_chunk_locked(final_chunk)
            self._finalized = True
            self._playback_cond.notify_all()

        # Немедленная остановка (barge-in / перезапуск запроса)
        if not graceful:
            self._stop_event.set()
            with self._lock:
                self._futures.clear()
                self._finalized = True
                self._playback_cond.notify_all()
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)

        if not graceful:
            try:
                sd.stop()
            except Exception:
                pass

        if self._playback_thread and self._playback_thread.is_alive():
            if graceful:
                self._playback_thread.join(timeout=60.0)
                if self._playback_thread.is_alive():
                    kiwi_log("STREAM-TTS", "Playback thread did not finish in 60s, forcing stop", level="WARNING")
                    self._stop_event.set()
                    self._playback_thread.join(timeout=3.0)
            else:
                self._playback_thread.join(timeout=2.0)

        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=not graceful)
            self._executor = None

        if graceful:
            kiwi_log("STREAM-TTS", "Manager stopped (graceful)", level="INFO")
        else:
            kiwi_log("STREAM-TTS", "Manager stopped (immediate)", level="INFO")

    def _clean_token(self, token: str) -> str:
        """Очищает токен от JSON-паттернов delta content.

        ЧИСТЫЙ REGEX — без ast.literal_eval для избежания проблем с форматированием.

        Обрабатывает случаи:
        - {'type': 'text', 'text': '...'} (одиночный dict)
        - Конкатенацию нескольких dict'ов
        """
        if not isinstance(token, str):
            return str(token) if token else ""

        stripped = token.strip()
        if not stripped:
            return ""

        # Если это обычный текст без паттернов dict — возвращаем как есть
        if not (("'text'" in stripped or '"text"' in stripped) and
                (stripped.startswith('{') or stripped.startswith('['))):
            return token

        # Случай 1: Список с dict'ами [{'type': 'text', 'text': '...'}]
        if stripped.startswith('[') and stripped.endswith(']'):
            # Ищем все dict'ы внутри списка: {...}
            dict_matches = re.findall(r'\{[^{}]*\}', stripped)
            if dict_matches:
                texts = []
                for dict_str in dict_matches:
                    # Ищем 'text': '...' или "text": "..."
                    text_match = re.search(r"'text':\s*'([^']*?)'", dict_str)
                    if text_match:
                        texts.append(text_match.group(1))
                    else:
                        text_match = re.search(r'"text":\s*"([^"]*?)"', dict_str)
                        if text_match:
                            texts.append(text_match.group(1))
                if texts:
                    return "".join(texts)

        # Случай 2: Одиночный dict — ищем 'text': '...' или "text": "..."
        if stripped.startswith('{') and stripped.endswith('}'):
            text_match = re.search(r"'text':\s*'([^']*?)'", stripped)
            if text_match:
                return text_match.group(1)
            text_match = re.search(r'"text":\s*"([^"]*?)"', stripped)
            if text_match:
                return text_match.group(1)

        # Случай 3: Конкатенация dict'ов — ищем все вхождения
        matches = re.findall(r"'text':\s*'([^']*?)'", token)
        if matches:
            result = "".join(matches)
            if result:
                return result

        matches = re.findall(r'"text":\s*"([^"]*?)"', token)
        if matches:
            result = "".join(matches)
            if result:
                return result

        # Случай 4: Разбиваем по }{ и ищем text в каждой части
        if '}{' in token:
            parts = token.split('}{')
            texts = []
            for i, part in enumerate(parts):
                # Добавляем скобки обратно
                if i == 0:
                    part = part + '}'
                elif i == len(parts) - 1:
                    part = '{' + part
                else:
                    part = '{' + part + '}'

                # Ищем text с помощью regex
                text_match = re.search(r"'text':\s*'([^']*?)'", part)
                if text_match:
                    texts.append(text_match.group(1))
                else:
                    text_match = re.search(r'"text":\s*"([^"]*?)"', part)
                    if text_match:
                        texts.append(text_match.group(1))

            if texts:
                return "".join(texts)

        return token

    def on_token(self, token: str):
        """Принимает токен от LLM и накапливает."""
        if not self._is_active:
            return

        # Очищаем токен от JSON-мусора перед добавлением в буфер
        cleaned_token = self._clean_token(token)
        if not cleaned_token:
            return

        with self._lock:
            self._buffer += cleaned_token
            self._try_send_chunk()

    def _try_send_chunk(self):
        """Пытается отправить накопленный чанк в TTS."""
        available = self._buffer[len(self._sent_text):]

        # Ищем конец предложения
        sentence_end = -1
        for i, char in enumerate(available):
            if char in '.!?' and i > self.min_chunk_chars:
                sentence_end = i + 1
                break
            # Или запятую после достаточного количества символов
            if char == ',' and i > self.max_chunk_chars:
                sentence_end = i + 1
                break

        # Если накопилось много текста без знаков препинания — разбиваем по пробелу
        if sentence_end == -1 and len(available) > self.max_chunk_chars:
            # Ищем последний пробел перед max_chunk_chars
            last_space = available.rfind(' ', self.min_chunk_chars, self.max_chunk_chars)
            if last_space > 0:
                sentence_end = last_space

        if sentence_end > 0:
            chunk = available[:sentence_end].strip()
            if chunk:
                self._enqueue_chunk_locked(chunk)
                self._sent_text = self._buffer[:len(self._sent_text) + sentence_end]
                kiwi_log("STREAM-TTS", f"Queued chunk ({len(chunk)} chars): {chunk[:50]}...", level="INFO")

    def _enqueue_chunk_locked(self, chunk: str):
        """Ставит чанк в параллельную генерацию (вызывать только под self._lock)."""
        if not self._executor:
            return
        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        self._futures[chunk_id] = self._executor.submit(self._synthesize_job, chunk_id, chunk)
        self._playback_cond.notify_all()

    def _synthesize_job(self, chunk_id: int, chunk: str):
        """Генерирует аудио для чанка (в пуле потоков)."""
        kiwi_log("STREAM-TTS", f"Processing chunk #{chunk_id}: {chunk[:60]}...", level="INFO")
        if self.tts_synthesize_callback:
            return self.tts_synthesize_callback(chunk)
        # Fallback: старый путь, когда один callback делает и синтез, и проигрывание.
        return chunk

    def _playback_worker(self):
        """Воспроизводит результаты строго в порядке исходных чанков."""
        while True:
            try:
                with self._playback_cond:
                    while True:
                        if self._stop_event.is_set() and not self._graceful_shutdown:
                            kiwi_log("STREAM-TTS", "Playback worker stopped by event", level="INFO")
                            return

                        future = self._futures.get(self._next_play_id)
                        if future is not None:
                            chunk_id = self._next_play_id
                            break

                        if self._finalized and self._next_play_id >= self._next_chunk_id:
                            kiwi_log("STREAM-TTS", "Playback worker completed all chunks", level="INFO")
                            return

                        self._playback_cond.wait(timeout=0.2)

                result = None
                wait_started = time.time()
                while True:
                    if self._stop_event.is_set() and not self._graceful_shutdown:
                        kiwi_log("STREAM-TTS", "Playback worker interrupted while waiting synthesis", level="INFO")
                        return
                    try:
                        result = future.result(timeout=0.2)
                        break
                    except FuturesTimeoutError:
                        if (time.time() - wait_started) >= self.max_chunk_wait_s:
                            kiwi_log(
                                "STREAM-TTS",
                                f"Synthesis timeout in chunk #{chunk_id} after {self.max_chunk_wait_s:.1f}s",
                                level="WARNING",
                            )
                            try:
                                future.cancel()
                            except Exception:
                                pass
                            result = None
                            break
                        continue
                    except Exception as e:
                        kiwi_log("STREAM-TTS", f"Synthesis error in chunk #{chunk_id}: {e}", level="ERROR")
                        result = None
                        break

                with self._playback_cond:
                    self._futures.pop(chunk_id, None)
                    self._next_play_id += 1
                    self._playback_cond.notify_all()

                if self._stop_event.is_set() and not self._graceful_shutdown:
                    kiwi_log("STREAM-TTS", "Playback worker stopped before playback", level="INFO")
                    return

                if self.tts_synthesize_callback and self.playback_callback:
                    if result and isinstance(result, tuple) and len(result) == 2:
                        audio, sample_rate = result
                        if audio is not None and len(audio) > 0:
                            self.playback_callback(audio, sample_rate)
                else:
                    if isinstance(result, str) and result.strip():
                        self.tts_callback(result)
            except Exception as e:
                kiwi_log("STREAM-TTS", f"Error in playback worker: {e}", level="ERROR")
