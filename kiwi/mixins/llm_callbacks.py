"""LLM callback mixin — token, activity, and completion handlers."""

import time

from kiwi.state_machine import DialogueState
from kiwi.utils import kiwi_log


class LLMCallbacksMixin:
    """WebSocket callback handlers for streaming LLM responses."""

    def _on_llm_token(self, token: str):
        """Callback on each token from the LLM (WebSocket delta event)."""
        if token:
            with self._stream_watchdog_lock:
                self._stream_watchdog_first_token_seen = True
                self._stream_watchdog_token_count += 1
                self._stream_watchdog_total_chars += len(token)
                self._stream_watchdog_last_token_ts = time.time()

        if self._streaming_tts_manager:
            self._streaming_tts_manager.on_token(token)

        if self._task_status_announcer and hasattr(self.openclaw, '_accumulated_text'):
            self._task_status_announcer.on_text_update(self.openclaw._accumulated_text)

    def _on_agent_activity(self, activity: dict):
        """Callback for intermediate agent activity (tool/lifecycle)."""
        with self._stream_watchdog_lock:
            self._stream_watchdog_last_activity_ts = time.time()

        if not self._task_status_announcer:
            return
        message = activity.get("message", "")
        if not message:
            return
        self._task_status_announcer.on_activity(message)

    def _on_llm_complete(self, full_text: str):
        """Callback when LLM generation is complete (WebSocket final event)."""
        self._stop_stream_watchdog()

        with self._streaming_completion_lock:
            if self._streaming_generation == 0:
                kiwi_log("LLM", "Duplicate/stale completion callback ignored", level="WARNING")
                return
            self._streaming_generation = 0

        # Fallback: use accumulated text if full_text is empty
        if not full_text and hasattr(self.openclaw, '_accumulated_text'):
            accumulated = self.openclaw._accumulated_text
            if accumulated:
                kiwi_log("LLM", f"Generation complete with EMPTY final, using accumulated: {len(accumulated)} chars", level="WARNING")
                full_text = accumulated
            else:
                kiwi_log("LLM", "Generation complete: 0 chars (EMPTY)", level="WARNING")
        else:
            kiwi_log("LLM", f"Generation complete: {len(full_text)} chars", level="INFO")

        if self._task_status_announcer:
            self._task_status_announcer.stop()
            self._task_status_announcer = None

        if self._streaming_tts_manager:
            self._streaming_tts_manager.stop()
            self._streaming_tts_manager = None

        # Safety fallback: if no streaming chunk was played, speak full text
        if (not self._streaming_response_playback_started) and full_text and full_text.strip():
            kiwi_log("STREAM-TTS", "No playback started for this response, using fallback speak()", level="WARNING")
            self.speak(full_text, style=self._streaming_style)
            if not self._barge_in_requested:
                self._start_idle_timer()
            return

        # Post-playback transition (mirrors _play_audio_interruptible)
        self._is_speaking = False
        self.listener._tts_start_time = time.time()
        self.listener.activate_dialog_mode()

        if self.listener.dialog_mode:
            self._set_state(DialogueState.LISTENING)
        else:
            self._set_state(DialogueState.IDLE)

        if not self._barge_in_requested:
            self._start_idle_timer()
