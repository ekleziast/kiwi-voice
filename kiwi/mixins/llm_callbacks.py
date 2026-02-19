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
                first_token = not self._stream_watchdog_first_token_seen
                self._stream_watchdog_first_token_seen = True
                self._stream_watchdog_token_count += 1
                self._stream_watchdog_total_chars += len(token)
                self._stream_watchdog_last_token_ts = time.time()

            # Extend THINKING state timeout while tokens are arriving.
            try:
                with self._state_lock:
                    if self._dialogue_state == DialogueState.THINKING:
                        timeout = self._state_timeouts.get(DialogueState.THINKING, 60.0)
                        self._state_until = time.time() + timeout
            except Exception:
                pass

            # Stop the status announcer on first real text — response is streaming,
            # no need for "Думаю над ответом..." status messages anymore.
            # Use stop_nowait() to avoid blocking LLM token delivery for 2s
            # while the announcer thread finishes its REST TTS call.
            if first_token and self._task_status_announcer:
                self._task_status_announcer.stop_nowait()
                self._task_status_announcer = None

        if self._streaming_tts_manager:
            self._streaming_tts_manager.on_token(token)

        if self._task_status_announcer and hasattr(self.openclaw, '_accumulated_text'):
            self._task_status_announcer.on_text_update(self.openclaw._accumulated_text)

    def _on_agent_activity(self, activity: dict):
        """Callback for intermediate agent activity (tool/lifecycle)."""
        with self._stream_watchdog_lock:
            self._stream_watchdog_last_activity_ts = time.time()

        # Extend THINKING state timeout — agent IS working (tool calls, etc.).
        # Without this, the 60s hard timeout kills long-running tasks like
        # model downloads or code generation.
        try:
            with self._state_lock:
                if self._dialogue_state == DialogueState.THINKING:
                    timeout = self._state_timeouts.get(DialogueState.THINKING, 60.0)
                    self._state_until = time.time() + timeout
        except Exception:
            pass

        message = activity.get("message", "")

        # Flush ElevenLabs WS buffer — agent switched from text to tool work,
        # the last word of the text segment is stuck in the buffer without a
        # trailing space.  flush_wave() is idempotent (no-op if buffer empty).
        if self._streaming_tts_manager and hasattr(self._streaming_tts_manager, 'flush_wave'):
            self._streaming_tts_manager.flush_wave()

        # Restart announcer if it was killed by first token but agent is still
        # doing tool work within the same wave (no lifecycle:end yet).
        if not self._task_status_announcer and self._streaming_tts_manager and message:
            kiwi_log("LLM", "Agent doing tool work — restarting status announcer", level="INFO")
            self._create_status_announcer(message, intervals=[5, 15, 30, 60, 120])

        if not self._task_status_announcer or not message:
            return
        self._task_status_announcer.on_activity(message)

    def _on_llm_resume(self):
        """Agent continues after lifecycle:end — restart status announcer.

        Called via on_resume callback when deferred final is cancelled,
        meaning the agent is doing more work (tool calls, research steps)
        after what initially looked like the end of the response.
        """
        # Only restart if we're in a streaming session and announcer is dead
        if not self._streaming_tts_manager:
            return
        if self._task_status_announcer is not None:
            return

        kiwi_log("LLM", "Agent continues — restarting status announcer for inter-wave updates", level="INFO")
        self._create_status_announcer(
            "продолжаю выполнение",
            intervals=[5, 15, 30, 60, 120],
        )

    def _on_wave_end(self):
        """lifecycle:end arrived — flush ElevenLabs WS buffer between waves."""
        if self._streaming_tts_manager and hasattr(self._streaming_tts_manager, 'flush_wave'):
            self._streaming_tts_manager.flush_wave()

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

        # Safety fallback: speak full text only if streaming never started playback.
        # If playback started, audio was delivered — even if the WS later died
        # (idle timeout between waves), the graceful stop drains the queue.
        if not self._streaming_response_playback_started and full_text and full_text.strip():
            kiwi_log("STREAM-TTS", "No playback started for this response, using fallback speak()", level="WARNING")
            self.speak(full_text, style=self._streaming_style)
            if not self._barge_in_requested:
                self._start_idle_timer()
            return

        # Post-playback transition (mirrors _play_audio_interruptible)
        self._is_speaking = False
        self.listener._tts_start_time = time.time()
        self.listener._last_tts_text = full_text or ""
        self.listener._last_tts_time = time.time()
        self.listener.activate_dialog_mode()

        if self.listener.dialog_mode:
            self._set_state(DialogueState.LISTENING)
        else:
            self._set_state(DialogueState.IDLE)

        if not self._barge_in_requested:
            self._start_idle_timer()
