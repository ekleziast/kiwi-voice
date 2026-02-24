#!/usr/bin/env python3
"""Task status announcer for Kiwi Voice."""

import threading
import time
from typing import Callable, Optional

from kiwi.utils import kiwi_log
from kiwi.i18n import t


class TaskStatusAnnouncer:
    """Announces the status of long-running tasks based on LLM context.

    Monitors streaming text from the LLM and periodically speaks
    brief status updates so the user knows what is happening.
    """

    def __init__(self, speak_func: Callable, intervals: list = None):
        """
        Args:
            speak_func: Function for speech output (_speak_chunk)
            intervals: Notification intervals in seconds [10, 30, 60, 120, 180]
        """
        self.speak_func = speak_func
        self.intervals = intervals or [6, 20, 45, 90, 150]

        self._start_time = 0.0
        self._last_text = ""
        self._last_announce_time = 0.0
        self._announced_intervals = set()  # Which intervals have already been announced
        self._pending_activity = ""
        self._last_activity_key = ""
        self._last_activity_time = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tts_is_playing = False
        self._lock = threading.Lock()

        self._command = ""                 # original voice command for context
        self._announced_text_len = 0       # track text already consumed by announcer
        self._last_spoken_status = ""      # prevent exact same message twice in a row

        # Minimum interval between announcements (spam protection)
        self._min_interval = 8.0
        self._activity_min_interval = 4.0

    def start(self, command: str):
        """Start task monitoring."""
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
            self._command = command
            self._announced_text_len = 0
            self._last_spoken_status = ""

        # Start background monitoring thread
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        kiwi_log("STATUS-ANNOUNCER", f"Started monitoring for: {command[:50]}...", level="INFO")

    def on_text_update(self, accumulated_text: str):
        """Called on each delta from the LLM."""
        with self._lock:
            self._last_text = accumulated_text

    def on_tts_playing(self, is_playing: bool):
        """Main TTS started/stopped playback."""
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
        """Receive and speak an execution step as soon as possible."""
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
        """Stop monitoring (blocking -- waits for the thread to finish)."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        kiwi_log("STATUS-ANNOUNCER", "Stopped", level="INFO")

    def stop_nowait(self):
        """Stop monitoring without waiting (non-blocking).

        Used from _on_llm_token, where a 2s block is unacceptable
        because it delays all subsequent LLM tokens.
        """
        self._stop_event.set()
        kiwi_log("STATUS-ANNOUNCER", "Stop requested (nowait)", level="INFO")

    def _monitor_loop(self):
        """Background monitoring thread."""
        while not self._stop_event.is_set():
            try:
                time.sleep(1.0)  # Check every second
                self._announce_pending_activity()

                elapsed = time.time() - self._start_time

                # Check if a status announcement is needed
                for interval in self.intervals:
                    if elapsed >= interval and interval not in self._announced_intervals:
                        # Check minimum interval since last announcement
                        time_since_last = time.time() - self._last_announce_time
                        if time_since_last < self._min_interval:
                            continue

                        # Don't announce if main TTS is playing
                        with self._lock:
                            if self._tts_is_playing:
                                continue

                            # Generate status message
                            status_msg = self._generate_status_message(elapsed)

                        if status_msg:
                            kiwi_log("STATUS-ANNOUNCER", f"Announcing at {elapsed:.0f}s: {status_msg[:50]}...", level="INFO")
                            self.speak_func(status_msg)

                            with self._lock:
                                self._announced_intervals.add(interval)
                                self._last_announce_time = time.time()

                        break  # Announced one interval, wait for the next

            except Exception as e:
                kiwi_log("STATUS-ANNOUNCER", f"Error in monitor loop: {e}", level="ERROR")

    def _announce_pending_activity(self):
        """Speak pending activity if conditions are appropriate."""
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
        """Generate a status message based on context.

        IMPORTANT: caller must hold self._lock already.

        Logic:
        1. Check for NEW text (beyond _announced_text_len)
        2. If new text found → extract last sentence → use if informative
        3. If no new text → use index-based varied fallbacks (unique per interval)
        4. Never repeat the exact same message twice in a row
        """
        text = self._last_text
        new_text = ""

        if text and len(text) > self._announced_text_len:
            new_text = text[self._announced_text_len:]
            self._announced_text_len = len(text)

        # Try to extract an informative sentence from new text
        if new_text and len(new_text.strip()) >= 20:
            # Split on sentence boundaries
            for sep in ['. ', '! ', '? ', '\n']:
                parts = new_text.rsplit(sep, 1)
                if len(parts) > 1:
                    new_text = parts[-1]

            candidate = new_text.strip().rstrip('.')
            if len(candidate) > 80:
                candidate = candidate[:60].rsplit(' ', 1)[0] + "..."

            if len(candidate) >= 20 and candidate != self._last_spoken_status:
                self._last_spoken_status = candidate
                return candidate

        # Fallback: index-based varied messages — each interval gets a unique one
        fallbacks = [
            t("status.thinking"),
            t("status.processing"),
            t("status.working"),
            t("status.continuing"),
            t("status.almost_done"),
        ]

        idx = len(self._announced_intervals) % len(fallbacks)
        candidate = fallbacks[idx]

        # If this would repeat the last spoken status, advance to next variant
        if candidate == self._last_spoken_status:
            idx = (idx + 1) % len(fallbacks)
            candidate = fallbacks[idx]

        self._last_spoken_status = candidate
        return candidate
