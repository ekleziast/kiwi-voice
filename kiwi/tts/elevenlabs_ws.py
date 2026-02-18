#!/usr/bin/env python3
"""ElevenLabs WebSocket Input Streaming for Kiwi Voice.

Maintains a single persistent WS connection and streams text tokens
directly as they arrive from the LLM, receiving audio back in real-time.
"""

import base64
import json
import re
import struct
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from kiwi.utils import kiwi_log

# PCM output: 16-bit LE mono at 24kHz
_WS_OUTPUT_FORMAT = "pcm_24000"
_WS_SAMPLE_RATE = 24000


class ElevenLabsWSStreamManager:
    """WebSocket-based streaming TTS manager for ElevenLabs.

    Same interface as StreamingTTSManager: start(), on_token(), stop().
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str,
        voice_settings: Dict[str, Any],
        playback_callback: Callable[[np.ndarray, int], None],
        speed: float = 1.0,
        inactivity_timeout: float = 60.0,
        playback_buffer_s: float = 1.0,
    ):
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id
        self._voice_settings = dict(voice_settings)
        self._speed = speed
        self._playback_callback = playback_callback
        self._inactivity_timeout = inactivity_timeout
        self._playback_buffer_s = max(0.2, float(playback_buffer_s))

        self._ws = None
        self._buffer = ""
        self._lock = threading.Lock()
        self._is_active = False
        self._stop_event = threading.Event()
        self._recv_thread: Optional[threading.Thread] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._audio_queue: Optional[Any] = None
        self._ws_connected = False
        self._eos_sent = False
        self._is_final_received = False

    # ------------------------------------------------------------------
    # Token cleaning (same logic as StreamingTTSManager._clean_token)
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_token(token: str) -> str:
        """Clean JSON delta content wrappers from LLM tokens."""
        if not isinstance(token, str):
            return str(token) if token else ""

        stripped = token.strip()
        if not stripped:
            return ""

        if not (("'text'" in stripped or '"text"' in stripped) and
                (stripped.startswith('{') or stripped.startswith('['))):
            return token

        if stripped.startswith('[') and stripped.endswith(']'):
            dict_matches = re.findall(r'\{[^{}]*\}', stripped)
            if dict_matches:
                texts = []
                for dict_str in dict_matches:
                    text_match = re.search(r"'text':\s*'([^']*?)'", dict_str)
                    if text_match:
                        texts.append(text_match.group(1))
                    else:
                        text_match = re.search(r'"text":\s*"([^"]*?)"', dict_str)
                        if text_match:
                            texts.append(text_match.group(1))
                if texts:
                    return "".join(texts)

        if stripped.startswith('{') and stripped.endswith('}'):
            text_match = re.search(r"'text':\s*'([^']*?)'", stripped)
            if text_match:
                return text_match.group(1)
            text_match = re.search(r'"text":\s*"([^"]*?)"', stripped)
            if text_match:
                return text_match.group(1)

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

        if '}{' in token:
            parts = token.split('}{')
            texts = []
            for i, part in enumerate(parts):
                if i == 0:
                    part = part + '}'
                elif i == len(parts) - 1:
                    part = '{' + part
                else:
                    part = '{' + part + '}'
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open WS connection, start recv + playback threads."""
        import queue as _queue
        import websocket as _websocket

        self._stop_event.clear()
        self._buffer = ""
        self._eos_sent = False
        self._is_final_received = False
        self._audio_queue = _queue.Queue()

        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech"
            f"/{self._voice_id}/stream-input"
            f"?model_id={self._model_id}"
            f"&output_format={_WS_OUTPUT_FORMAT}"
        )

        kiwi_log("ELEVENLABS-WS", f"Connecting to {url[:80]}...", level="INFO")
        try:
            self._ws = _websocket.WebSocket()
            self._ws.settimeout(self._inactivity_timeout)
            self._ws.connect(url)
            self._ws_connected = True
        except Exception as exc:
            kiwi_log("ELEVENLABS-WS", f"Connection failed: {exc}", level="ERROR")
            self._ws = None
            self._ws_connected = False
            return

        # Send Begin-of-Stream (BOS) message
        voice_settings = dict(self._voice_settings)
        if self._speed != 1.0:
            voice_settings["speed"] = self._speed
        bos = {
            "text": " ",
            "xi_api_key": self._api_key,
            "voice_settings": voice_settings,
            "generation_config": {
                "chunk_length_schedule": [50, 120, 200, 260],
            },
        }
        try:
            self._ws.send(json.dumps(bos))
            kiwi_log("ELEVENLABS-WS", "BOS sent", level="INFO")
        except Exception as exc:
            kiwi_log("ELEVENLABS-WS", f"Failed to send BOS: {exc}", level="ERROR")
            self._close_ws()
            return

        self._is_active = True

        self._recv_thread = threading.Thread(
            target=self._recv_worker, daemon=True, name="kiwi-11labs-ws-recv"
        )
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True, name="kiwi-11labs-ws-play"
        )
        self._recv_thread.start()
        self._playback_thread.start()
        kiwi_log("ELEVENLABS-WS", "Manager started", level="INFO")

    def on_token(self, token: str):
        """Accept an LLM token and forward to ElevenLabs WS."""
        if not self._is_active or not self._ws_connected:
            return

        cleaned = self._clean_token(token)
        if not cleaned:
            return

        with self._lock:
            self._buffer += cleaned
            self._flush_buffer()

    def stop(self, graceful: bool = True):
        """Stop the WS streaming manager.

        graceful=True: send remaining buffer + EOS, wait for isFinal and playback.
        graceful=False: close WS immediately (barge-in).
        """
        kiwi_log("ELEVENLABS-WS",
                  f"Stopping (graceful={graceful})", level="INFO")
        self._is_active = False

        if not graceful:
            self._stop_event.set()
            self._close_ws()
            if self._audio_queue:
                # Drain and signal end
                try:
                    while not self._audio_queue.empty():
                        self._audio_queue.get_nowait()
                except Exception:
                    pass
                self._audio_queue.put(None)
            if self._playback_thread and self._playback_thread.is_alive():
                self._playback_thread.join(timeout=2.0)
            if self._recv_thread and self._recv_thread.is_alive():
                self._recv_thread.join(timeout=2.0)
            kiwi_log("ELEVENLABS-WS", "Manager stopped (immediate)", level="INFO")
            return

        # Graceful: send remaining buffer + EOS
        with self._lock:
            remaining = self._buffer.strip()
            if remaining:
                self._send_text(remaining, flush=True)
                self._buffer = ""

        self._send_eos()

        # Wait for recv thread (isFinal) with timeout
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=30.0)
            if self._recv_thread.is_alive():
                kiwi_log("ELEVENLABS-WS",
                         "Recv thread did not finish in 30s, forcing", level="WARNING")
                self._stop_event.set()
                self._close_ws()
                self._recv_thread.join(timeout=3.0)

        # Wait for playback to drain
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=60.0)
            if self._playback_thread.is_alive():
                kiwi_log("ELEVENLABS-WS",
                         "Playback thread did not finish in 60s", level="WARNING")

        self._close_ws()
        kiwi_log("ELEVENLABS-WS", "Manager stopped (graceful)", level="INFO")

    # ------------------------------------------------------------------
    # Internal: buffer management
    # ------------------------------------------------------------------

    def _flush_buffer(self):
        """Send accumulated words to the WS. Call under self._lock."""
        buf = self._buffer

        # Find the last space — only send complete words
        last_space = buf.rfind(' ')
        if last_space <= 0:
            # Not enough for a complete word yet
            # But check for sentence-end on the whole buffer
            return

        to_send = buf[:last_space + 1]  # include trailing space
        self._buffer = buf[last_space + 1:]

        # Check if we should flush (sentence boundary)
        flush = bool(re.search(r'[.!?;:]\s*$', to_send))

        self._send_text(to_send, flush=flush)

    def _send_text(self, text: str, flush: bool = False):
        """Send a text message to the ElevenLabs WS."""
        if not self._ws_connected or self._eos_sent:
            return
        try:
            msg: Dict[str, Any] = {"text": text}
            if flush:
                msg["flush"] = True
            self._ws.send(json.dumps(msg))
            tag = " [flush]" if flush else ""
            kiwi_log("ELEVENLABS-WS",
                     f"Sent {len(text)} chars{tag}: {text[:60].strip()}",
                     level="DEBUG")
        except Exception as exc:
            kiwi_log("ELEVENLABS-WS", f"Send error: {exc}", level="ERROR")
            self._ws_connected = False

    def _send_eos(self):
        """Send End-of-Stream to signal we're done sending text."""
        if not self._ws_connected or self._eos_sent:
            return
        try:
            self._ws.send(json.dumps({"text": ""}))
            self._eos_sent = True
            kiwi_log("ELEVENLABS-WS", "EOS sent", level="INFO")
        except Exception as exc:
            kiwi_log("ELEVENLABS-WS", f"EOS send error: {exc}", level="ERROR")

    def _close_ws(self):
        """Close the WebSocket connection."""
        self._ws_connected = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    # ------------------------------------------------------------------
    # Recv thread
    # ------------------------------------------------------------------

    def _recv_worker(self):
        """Receive audio chunks from ElevenLabs WS."""
        chunks_received = 0
        try:
            while not self._stop_event.is_set():
                if not self._ws or not self._ws_connected:
                    break
                try:
                    raw = self._ws.recv()
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    kiwi_log("ELEVENLABS-WS",
                             f"Recv error: {exc}", level="ERROR")
                    break

                if not raw:
                    break

                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue

                # Check for final message
                if msg.get("isFinal"):
                    self._is_final_received = True
                    kiwi_log("ELEVENLABS-WS",
                             f"isFinal received after {chunks_received} chunks",
                             level="INFO")
                    break

                # Extract audio
                audio_b64 = msg.get("audio")
                if not audio_b64:
                    continue

                try:
                    audio_bytes = base64.b64decode(audio_b64)
                except Exception:
                    continue

                if not audio_bytes:
                    continue

                # PCM 16-bit LE -> float32
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                chunks_received += 1
                self._audio_queue.put((audio, _WS_SAMPLE_RATE))

        except Exception as exc:
            kiwi_log("ELEVENLABS-WS",
                     f"Recv worker error: {exc}", level="ERROR")
        finally:
            # Signal playback that no more audio is coming
            if self._audio_queue:
                self._audio_queue.put(None)
            kiwi_log("ELEVENLABS-WS",
                     f"Recv worker exited ({chunks_received} chunks received)",
                     level="INFO")

    # ------------------------------------------------------------------
    # Playback thread
    # ------------------------------------------------------------------

    def _playback_worker(self):
        """Accumulate audio into buffer, play in batches for smooth output.

        Two thresholds:
        - min_samples  (~1s): target batch size for normal playback
        - min_play     (~0.3s): absolute floor — never send anything shorter
          to sd.play() to avoid audible micro-fragments.  Only the very last
          piece of the stream is exempt from the floor.
        """
        import queue as _queue

        chunks_played = 0
        pending: list = []       # list of np arrays
        pending_samples = 0      # total samples in pending
        min_samples = int(self._playback_buffer_s * _WS_SAMPLE_RATE)
        min_play = int(0.3 * _WS_SAMPLE_RATE)  # 300ms hard floor
        stream_ended = False

        try:
            while True:
                if self._stop_event.is_set():
                    kiwi_log("ELEVENLABS-WS",
                             "Playback stopped by event", level="INFO")
                    break

                # --- drain queue aggressively --------------------------------
                # Block up to 0.1s for the first item only when the pending
                # buffer is empty; after that do non-blocking bulk drain so we
                # never sit idle with audio waiting in the queue.
                wait = 0.1 if not pending else 0.0
                drained = 0
                while True:
                    try:
                        item = self._audio_queue.get(timeout=wait)
                    except _queue.Empty:
                        break

                    if item is None:
                        stream_ended = True
                        break

                    audio, _sr = item
                    if audio is not None and len(audio) > 0:
                        pending.append(audio)
                        pending_samples += len(audio)
                        drained += 1

                    # After first successful get, switch to non-blocking
                    wait = 0.0

                    # Stop draining once we have enough for a batch
                    if pending_samples >= min_samples:
                        break

                # --- decide whether to play -----------------------------------
                if not pending:
                    if stream_ended:
                        break
                    continue

                enough = pending_samples >= min_samples
                is_final_flush = stream_ended

                if not enough and not is_final_flush:
                    # Not enough audio yet and stream still going — wait more
                    continue

                if self._stop_event.is_set():
                    break

                # Concatenate accumulated fragments into one array
                combined = np.concatenate(pending) if len(pending) > 1 else pending[0]
                pending.clear()
                pending_samples = 0

                # Guard against micro-fragments (< 300ms) mid-stream.
                # Keep them in the buffer to merge with the next batch.
                if len(combined) < min_play and not is_final_flush:
                    pending.append(combined)
                    pending_samples = len(combined)
                    continue

                try:
                    dur = len(combined) / _WS_SAMPLE_RATE
                    kiwi_log("ELEVENLABS-WS",
                             f"Playing buffered chunk ({dur:.2f}s)",
                             level="DEBUG")
                    self._playback_callback(combined, _WS_SAMPLE_RATE)
                    chunks_played += 1
                except Exception as exc:
                    kiwi_log("ELEVENLABS-WS",
                             f"Playback error: {exc}", level="ERROR")

                if is_final_flush:
                    break

        except Exception as exc:
            kiwi_log("ELEVENLABS-WS",
                     f"Playback worker error: {exc}", level="ERROR")
        finally:
            kiwi_log("ELEVENLABS-WS",
                     f"Playback worker exited ({chunks_played} chunks played)",
                     level="INFO")
