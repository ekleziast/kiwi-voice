#!/usr/bin/env python3
"""
Kiwi Voice Service - OpenClaw Integration.

Main orchestrator: KiwiServiceOpenClaw class + main() entry point.
Extracted modules: config_loader, state_machine, streaming_tts,
openclaw_ws, openclaw_cli, task_announcer.
"""

import os
import sys

def _setup_utf8_console_windows():
    """Force UTF-8 in Windows console to avoid mojibake in Cyrillic logs."""
    if sys.platform != "win32":
        return

    # Best-effort: switch console code pages to UTF-8.
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

    # Ensure Python writes UTF-8 to console streams.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

_setup_utf8_console_windows()

from kiwi import PROJECT_ROOT, LOGS_DIR

# Load .env before any os.getenv() calls
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Подавляем warning от torchcodec
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", module="pyannote")

# Import logging utilities
try:
    from kiwi.utils import kiwi_log, setup_crash_protection
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("[WARN] utils.py not found, using basic logging")

# Добавляем ffmpeg в PATH для pydub (если указан через KIWI_FFMPEG_PATH)
ffmpeg_path = os.getenv("KIWI_FFMPEG_PATH", "")
if ffmpeg_path and os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = ffmpeg_path + os.pathsep + os.environ.get('PATH', '')
    kiwi_log("INIT", f"Added ffmpeg to PATH: {ffmpeg_path}", level="INFO")

# DEBUG: Log to file (opt-in via KIWI_DEBUG)
if os.getenv("KIWI_DEBUG"):
    with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'w', encoding='utf-8') as f:
        f.write(f'[START] Python started: {sys.executable}\n')
        f.write(f'[START] Args: {sys.argv}\n')
        f.write(f'[START] CWD: {os.getcwd()}\n')

import threading
import time
import json
import re
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import queue

import numpy as np
import sounddevice as sd
from pydub import AudioSegment

if os.getenv("KIWI_DEBUG"):
    with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'a', encoding='utf-8') as f:
        f.write('[START] Imports done, loading modules...\n')

from kiwi.listener import KiwiListener, ListenerConfig
from kiwi.tts.runpod import TTSClient, TTSConfig
from kiwi.tts.piper import PiperTTS
from kiwi.tts.qwen_local import LocalQwenTTSClient, LocalQwenTTSConfig
from kiwi.tts.elevenlabs import ElevenLabsTTSClient, ElevenLabsTTSConfig

# Speaker Manager + Voice Security
try:
    from kiwi.speaker_manager import SpeakerManager, VoicePriority
    SPEAKER_MANAGER_AVAILABLE = True
except ImportError:
    SPEAKER_MANAGER_AVAILABLE = False
    kiwi_log("KIWI", "Speaker Manager not available", level="WARNING")

try:
    from kiwi.voice_security import VoiceSecurity
    VOICE_SECURITY_AVAILABLE = True
except ImportError:
    VOICE_SECURITY_AVAILABLE = False
    kiwi_log("KIWI", "Voice Security not available", level="WARNING")

# Event Bus
try:
    from kiwi.event_bus import EventBus, EventType, get_event_bus, publish, subscribe
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    kiwi_log("KIWI", "Event Bus not available", level="WARNING")

with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'a', encoding='utf-8') as f:
    f.write('[START] All modules imported OK\n')



from kiwi.config_loader import load_config_yaml, check_cuda_available, KiwiConfig
from kiwi.state_machine import DialogueState
from kiwi.tts.streaming import StreamingTTSManager
from kiwi.openclaw_ws import OpenClawWebSocket
from kiwi.openclaw_cli import OpenClawCLI
from kiwi.task_announcer import TaskStatusAnnouncer
from kiwi.text_processing import clean_chunk_for_tts, normalize_tts_text, split_text_into_chunks


@dataclass
class CommandContext:
    """Per-invocation state carried between pipeline stages in _on_wake_word."""
    command: str
    command_lower: str = ""
    timestamp: float = 0.0
    speaker_id: str = "unknown"
    speaker_name: str = "Незнакомец"
    speaker_confidence: float = 0.0
    speaker_music_prob: float = 0.0
    is_owner: bool = False
    owner_profile_ready: bool = False
    approved_command_from_owner: Optional[str] = None
    abort: bool = False


class KiwiServiceOpenClaw:
    """Главный сервис голосового ассистента Киви с OpenClaw интеграцией."""
    
    _BEEP_FREQ = 880
    _BEEP_DURATION = 0.15
    _BEEP_SAMPLE_RATE = 44100
    _STARTUP_SAMPLE_RATE = 44100
    
    def __init__(self, config: Optional[KiwiConfig] = None):
        self.config = config or KiwiConfig()
        
        # Флаг: был ли уже отправлен системный промт
        self._system_prompt_sent = False
        
        # === STATE MACHINE ===
        self._dialogue_state = DialogueState.IDLE
        self._state_lock = threading.Lock()
        self._last_state_change = 0.0
        
        # Таймауты для каждого состояния
        self._state_timeouts = {
            DialogueState.IDLE: None,           # Бесконечный
            DialogueState.LISTENING: 5.0,      # 5 сек на команду
            DialogueState.PROCESSING: 25.0,     # LLM completeness/intent + buffer
            DialogueState.THINKING: 150.0,      # OpenClaw chat + buffer
            DialogueState.SPEAKING: None,       # До конца TTS
        }
        self._state_until = 0.0
        
        self._last_command = ""
        self._last_command_time = 0.0
        self._command_cooldown = 3.0
        self._last_beep_time = 0.0
        
        self._pending_phrase = ""
        self._pending_timestamp = 0.0
        self._pending_timeout = 8.0

        # Owner approval for third-party task execution
        self._owner_id = "owner"
        self._owner_name = getattr(self.config, "owner_name", "Owner")
        self._owner_approval_timeout = 120.0
        self._pending_owner_approval: Optional[Dict[str, Any]] = None
        self._owner_profile_warning_shown = False
        
        self._self_profile_created = False
        
        self._is_speaking = False
        self._barge_in_requested = False
        self._current_playback_thread: Optional[threading.Thread] = None
        
        self._idle_timer: Optional[threading.Timer] = None
        self._idle_delay = 1.5
        
        self._had_actual_command = False
        
        # === STREAMING TTS ===
        self._audio_queue: queue.Queue = queue.Queue()
        self._streaming_playback_thread: Optional[threading.Thread] = None
        self._streaming_stop_event = threading.Event()
        self._sd_play_lock = threading.Lock()
        self._is_streaming = False
        
        # === STREAMING TTS MANAGER ===
        self._streaming_tts_manager: Optional[StreamingTTSManager] = None
        self._current_streaming_response: str = ""
        self._streaming_style: str = "neutral"
        self._streaming_response_playback_started = False
        self._streaming_completion_lock = threading.Lock()
        self._streaming_generation = 0
        self._stream_watchdog_stop_event = threading.Event()
        self._stream_watchdog_thread: Optional[threading.Thread] = None
        self._stream_watchdog_lock = threading.Lock()
        self._stream_watchdog_last_token_ts = 0.0
        self._stream_watchdog_last_activity_ts = 0.0
        self._stream_watchdog_first_token_seen = False
        self._stream_watchdog_token_count = 0
        self._stream_watchdog_total_chars = 0
        self._stream_watchdog_command = ""
        self._stream_watchdog_retry_count = 0
        self._stream_watchdog_retrying = False
        
        # === TASK STATUS ANNOUNCER ===
        self._task_status_announcer: Optional[TaskStatusAnnouncer] = None
        
        with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'a', encoding='utf-8') as f:
            f.write('[START] KiwiServiceOpenClaw.__init__ starting...\n')
        
        self._beep_sound, self._beep_sr = self._generate_beep()
        self._startup_sound, self._startup_sr = self._generate_startup_sound()
        self._idle_sound, self._idle_sr = self._load_idle_sound()
        kiwi_log("SOUND", f"Loaded: confirmation={len(self._beep_sound)/self._beep_sr:.2f}s, startup={len(self._startup_sound)/self._startup_sr:.2f}s, idle={len(self._idle_sound)/self._idle_sr:.2f}s", level="INFO")
        
        with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'a', encoding='utf-8') as f:
            f.write('[START] Beep generated\n')
        
        # Инициализация OpenClaw: WebSocket (если enabled) или CLI
        self.openclaw = self._init_openclaw()
        
        with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'a', encoding='utf-8') as f:
            f.write('[START] OpenClaw initialized\n')
        
        # Инициализация TTS
        self.tts_provider = self.config.tts_provider
        self.tts_qwen_backend = self.config.tts_qwen_backend
        self.use_local_tts = self.config.use_local_tts

        if self.tts_provider == "piper":
            self.tts = PiperTTS(model_path=self.config.tts_piper_model_path)
            self.use_local_tts = True
            kiwi_log("TTS", "Initialized Piper TTS (local)", level="INFO")
        elif self.tts_provider == "elevenlabs":
            elevenlabs_config = ElevenLabsTTSConfig(
                api_key=self.config.tts_elevenlabs_api_key,
                default_voice_id=self.config.tts_elevenlabs_voice_id,
                model_id=self.config.tts_elevenlabs_model_id,
                output_format=self.config.tts_elevenlabs_output_format,
                timeout=self.config.tts_timeout,
                use_streaming_endpoint=self.config.tts_elevenlabs_use_streaming_endpoint,
                optimize_streaming_latency=self.config.tts_elevenlabs_optimize_streaming_latency,
                stability=self.config.tts_elevenlabs_stability,
                similarity_boost=self.config.tts_elevenlabs_similarity_boost,
                style=self.config.tts_elevenlabs_style,
                use_speaker_boost=self.config.tts_elevenlabs_use_speaker_boost,
                speed=self.config.tts_elevenlabs_speed,
                style_presets=self.config.tts_elevenlabs_style_presets,
            )
            self.tts = ElevenLabsTTSClient(elevenlabs_config)
            self.use_local_tts = False
            kiwi_log("TTS", f"Initialized ElevenLabs ({self.config.tts_elevenlabs_model_id})", level="INFO")
            kiwi_log(
                "TTS",
                f"ElevenLabs voice_id: {self.config.tts_elevenlabs_voice_id}, "
                f"stream endpoint: {self.config.tts_elevenlabs_use_streaming_endpoint}",
                level="INFO",
            )
        elif self.tts_provider == "qwen3" and self.tts_qwen_backend == "local":
            qwen_local_config = LocalQwenTTSConfig(
                model_size=self.config.tts_model_size,
                model_path=self.config.tts_local_model_path,
                tokenizer_path=self.config.tts_local_tokenizer_path,
                default_voice=self.config.tts_voice,
                device=self.config.tts_qwen_device,
            )
            self.tts = LocalQwenTTSClient(qwen_local_config)
            self.use_local_tts = True
            kiwi_log("TTS", f"Initialized Qwen3-TTS {self.config.tts_model_size} (local)", level="INFO")
            kiwi_log(
                "TTS",
                f"Local device: {self.tts.runtime_device.upper()} "
                f"(configured: {self.config.tts_qwen_device.upper()})",
                level="INFO",
            )
        else:
            tts_config = TTSConfig(
                endpoint_id=self.config.tts_endpoint_id,
                api_key=self.config.tts_api_key,
                default_voice=self.config.tts_voice,
                model_size=self.config.tts_model_size,
                timeout=self.config.tts_timeout,
                poll_interval=self.config.tts_poll_interval,
            )
            self.tts = TTSClient(tts_config)
            self.use_local_tts = False
            kiwi_log("TTS", f"Initialized Qwen3-TTS {self.config.tts_model_size} via RunPod", level="INFO")
        
        # Инициализация Listener
        listener_config = ListenerConfig(
            model_name=self.config.stt_model,
            device=self.config.stt_device,
            compute_type=self.config.stt_compute_type,
            wake_word=self.config.wake_word_keyword,
            position_limit=self.config.wake_word_position_limit,
        )
        self.listener = KiwiListener(
            config=listener_config, 
            on_wake_word=self._on_wake_word,
        )
        if getattr(self.listener, "speaker_manager", None) is not None:
            self._owner_id = getattr(self.listener.speaker_manager, "OWNER_ID", self._owner_id)
            # Propagate configured owner name to speaker manager
            self.listener.speaker_manager.OWNER_NAME = self._owner_name
        kiwi_log("LISTENER", "Initialized Kiwi Listener", level="INFO")
        
        # Состояние сервиса
        self.is_running = False
    
    def _init_openclaw(self):
        """Инициализирует OpenClaw клиент: WebSocket или CLI с fallback."""
        # Если WebSocket включен в конфиге - пробуем сначала его
        if self.config.ws_enabled:
            try:
                ws_client = OpenClawWebSocket(
                    config=self.config,
                    on_token=self._on_llm_token,
                    on_complete=self._on_llm_complete,
                    on_activity=self._on_agent_activity,
                    log_func=kiwi_log if UTILS_AVAILABLE else print,
                )
                # Пытаемся подключиться
                if ws_client.connect():
                    kiwi_log("KIWI", f"WebSocket connected to {self.config.ws_host}:{self.config.ws_port}", "INFO")
                    self._use_websocket = True
                    return ws_client
                else:
                    kiwi_log("KIWI", "WebSocket connection failed, will fallback to CLI", "WARN")
                    ws_client.close()
            except Exception as e:
                kiwi_log("KIWI", f"WebSocket initialization error: {e}", "ERROR")
        
        # Fallback на CLI
        self._use_websocket = False
        kiwi_log("KIWI", "Using OpenClaw CLI mode (streaming TTS disabled)", "INFO")
        cli_client = OpenClawCLI(
            openclaw_bin=self.config.openclaw_bin,
            session_id=self.config.openclaw_session_id,
            agent=self.config.openclaw_agent,
            timeout=self.config.openclaw_timeout,
            model=self.config.llm_model,
            retry_max=self.config.llm_retry_max,
            retry_delays=self.config.llm_retry_delays,
        )
        kiwi_log("KIWI", f"CLI connected to session: {cli_client.session_key}", "INFO")
        return cli_client
    
    def _on_llm_token(self, token: str):
        """Callback при получении токена от LLM (WebSocket delta event)."""
        if token:
            with self._stream_watchdog_lock:
                self._stream_watchdog_first_token_seen = True
                self._stream_watchdog_token_count += 1
                self._stream_watchdog_total_chars += len(token)
                self._stream_watchdog_last_token_ts = time.time()

        if self._streaming_tts_manager:
            self._streaming_tts_manager.on_token(token)
        
        # Передаём накопленный текст в TaskStatusAnnouncer
        if self._task_status_announcer and hasattr(self.openclaw, '_accumulated_text'):
            self._task_status_announcer.on_text_update(self.openclaw._accumulated_text)

    def _on_agent_activity(self, activity: dict):
        """Callback для промежуточной активности агента (tool/lifecycle)."""
        with self._stream_watchdog_lock:
            self._stream_watchdog_last_activity_ts = time.time()

        if not self._task_status_announcer:
            return
        message = activity.get("message", "")
        if not message:
            return
        self._task_status_announcer.on_activity(message)

    def _reset_stream_watchdog_progress(self):
        with self._stream_watchdog_lock:
            self._stream_watchdog_first_token_seen = False
            self._stream_watchdog_token_count = 0
            self._stream_watchdog_total_chars = 0
            self._stream_watchdog_last_token_ts = time.time()
            self._stream_watchdog_last_activity_ts = time.time()

    def _start_stream_watchdog(self, command: str):
        timeout_s = float(self.config.llm_stream_stall_timeout)
        if timeout_s <= 0:
            return

        self._stop_stream_watchdog()
        with self._stream_watchdog_lock:
            self._stream_watchdog_command = command
            self._stream_watchdog_retry_count = 0
            self._stream_watchdog_retrying = False
            self._stream_watchdog_first_token_seen = False
            self._stream_watchdog_token_count = 0
            self._stream_watchdog_total_chars = 0
            self._stream_watchdog_last_token_ts = time.time()
            self._stream_watchdog_last_activity_ts = time.time()

        self._stream_watchdog_stop_event.clear()
        self._stream_watchdog_thread = threading.Thread(
            target=self._stream_watchdog_loop,
            daemon=True,
            name="kiwi-stream-watchdog",
        )
        self._stream_watchdog_thread.start()
        kiwi_log("STREAM-WATCHDOG", f"Started (stall timeout: {timeout_s:.1f}s)", level="INFO")

    def _stop_stream_watchdog(self):
        self._stream_watchdog_stop_event.set()
        thread = self._stream_watchdog_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._stream_watchdog_thread = None

    def _stream_watchdog_loop(self):
        timeout_s = max(5.0, float(self.config.llm_stream_stall_timeout))
        no_first_token_timeout = max(timeout_s * 3.0, 45.0)
        low_progress_timeout = max(timeout_s * 3.0, 36.0)
        post_progress_timeout = max(timeout_s * 2.5, 30.0)
        no_processing_since: Optional[float] = None
        while not self._stream_watchdog_stop_event.wait(0.25):
            if timeout_s <= 0:
                return
            if not self.openclaw.is_processing():
                if no_processing_since is None:
                    no_processing_since = time.time()
                elif (time.time() - no_processing_since) >= 2.0:
                    return
                continue
            no_processing_since = None

            with self._stream_watchdog_lock:
                first_token_seen = self._stream_watchdog_first_token_seen
                token_count = self._stream_watchdog_token_count
                total_chars = self._stream_watchdog_total_chars
                now = time.time()
                stalled_for = now - self._stream_watchdog_last_token_ts
                activity_stalled_for = now - self._stream_watchdog_last_activity_ts
                retrying = self._stream_watchdog_retrying

            if not first_token_seen:
                stall_reason = "before first token"
                effective_timeout = no_first_token_timeout
            elif token_count < 3 or total_chars < 16:
                stall_reason = "low token progress"
                effective_timeout = low_progress_timeout
            else:
                stall_reason = "no token progress"
                effective_timeout = post_progress_timeout

            # Если агент ещё активен (tool calls идут), даём больше времени
            if activity_stalled_for < 5.0:
                effective_timeout = max(effective_timeout, 90.0)

            if retrying or stalled_for < effective_timeout:
                continue

            self._handle_streaming_stall(stalled_for, stall_reason)

    def _get_accumulated_stream_text(self) -> str:
        """Возвращает накопленный текст из WS-клиента (best effort)."""
        if not hasattr(self.openclaw, "_accumulated_text"):
            return ""

        try:
            if hasattr(self.openclaw, "_buffer_lock"):
                with self.openclaw._buffer_lock:
                    text = getattr(self.openclaw, "_accumulated_text", "") or getattr(self.openclaw, "_full_response", "")
            else:
                text = getattr(self.openclaw, "_accumulated_text", "") or getattr(self.openclaw, "_full_response", "")
        except Exception:
            text = getattr(self.openclaw, "_accumulated_text", "") or getattr(self.openclaw, "_full_response", "")

        return str(text).strip()

    def _finalize_stalled_stream_from_accumulated(self, stalled_for: float, stall_reason: str) -> bool:
        """Пробует завершить зависший стрим уже накопленным текстом."""
        accumulated = self._get_accumulated_stream_text()
        if len(accumulated) < 40:
            return False

        kiwi_log(
            "STREAM-WATCHDOG",
            f"Stalled ({stall_reason}) for {stalled_for:.1f}s, "
            f"finalizing from accumulated text ({len(accumulated)} chars)",
            level="WARNING",
        )

        try:
            self.openclaw.cancel()
        except Exception as e:
            kiwi_log("STREAM-WATCHDOG", f"Cancel before finalize failed: {e}", level="ERROR")

        with self._streaming_completion_lock:
            self._streaming_generation += 1
        self._on_llm_complete(accumulated)
        return True

    def _handle_streaming_stall(self, stalled_for: float, stall_reason: str):
        # Если ответ уже достаточно накопился и озвучка началась, безопаснее
        # завершить из накопленного текста, чем делать abort+retry и обрывать речь.
        accumulated_now = self._get_accumulated_stream_text()
        has_substantial_text = len(accumulated_now) >= 120
        if has_substantial_text and (
            self._streaming_response_playback_started or self._streaming_tts_manager is not None
        ):
            kiwi_log(
                "STREAM-WATCHDOG",
                f"Stall ({stall_reason}) for {stalled_for:.1f}s with "
                f"{len(accumulated_now)} accumulated chars; finalizing instead of retry",
                level="WARNING",
            )
            if self._finalize_stalled_stream_from_accumulated(stalled_for, stall_reason):
                self._stop_stream_watchdog()
                return

        retry_exhausted = False
        with self._stream_watchdog_lock:
            if self._stream_watchdog_retrying:
                return
            max_retries = max(0, int(self.config.llm_stream_stall_retry_max))
            if self._stream_watchdog_retry_count >= max_retries:
                retry_exhausted = True
            else:
                self._stream_watchdog_retrying = True
                self._stream_watchdog_retry_count += 1
                retry_no = self._stream_watchdog_retry_count
                command = self._stream_watchdog_command

        if retry_exhausted:
            if self._finalize_stalled_stream_from_accumulated(stalled_for, stall_reason):
                self._stop_stream_watchdog()
                return

            kiwi_log(
                "STREAM-WATCHDOG",
                f"Stalled ({stall_reason}) for {stalled_for:.1f}s, "
                "retry budget exhausted and no usable accumulated text",
                level="ERROR",
            )

            self._stop_stream_watchdog()
            self.openclaw.cancel()

            if self._streaming_tts_manager:
                self._streaming_tts_manager.stop(graceful=False)
                self._streaming_tts_manager = None
            if self._task_status_announcer:
                self._task_status_announcer.stop()
                self._task_status_announcer = None

            self._set_state(DialogueState.IDLE)
            self.speak("Ответ завис. Повтори, пожалуйста.", style="calm")
            return

        kiwi_log(
            "STREAM-WATCHDOG",
            f"Stall ({stall_reason}) for {stalled_for:.1f}s. "
            f"Retrying request ({retry_no}/{self.config.llm_stream_stall_retry_max})",
            level="WARNING",
        )

        try:
            self.openclaw.cancel()
            with self._streaming_completion_lock:
                self._streaming_generation += 1
            self._streaming_response_playback_started = False

            if self._streaming_tts_manager:
                self._streaming_tts_manager.stop(graceful=False)
                self._streaming_tts_manager = None
            if self._task_status_announcer:
                self._task_status_announcer.stop()
                self._task_status_announcer = None

            self._start_streaming_runtime(command)
            resend_ok = self.openclaw.send_message(command)
            if not resend_ok:
                kiwi_log("STREAM-WATCHDOG", "Retry send failed", level="ERROR")
                self._set_state(DialogueState.IDLE)
                self.speak("Не получилось получить ответ. Попробуй ещё раз.", style="calm")
                self._stop_stream_watchdog()
                return

            self._reset_stream_watchdog_progress()
            kiwi_log("STREAM-WATCHDOG", "Retry sent successfully", level="INFO")
        finally:
            with self._stream_watchdog_lock:
                self._stream_watchdog_retrying = False

    def _start_streaming_runtime(self, command: str):
        """Запускает инфраструктуру streaming TTS для текущего запроса."""
        if self._streaming_tts_manager:
            kiwi_log("KIWI", "Stopping previous StreamingTTSManager", level="INFO")
            self._streaming_tts_manager.stop(graceful=False)
            self._streaming_tts_manager = None
        if self._task_status_announcer:
            kiwi_log("KIWI", "Stopping previous TaskStatusAnnouncer", level="INFO")
            self._task_status_announcer.stop()
            self._task_status_announcer = None

        synthesis_workers = 2 if self.tts_provider == "elevenlabs" else 1
        self._streaming_tts_manager = StreamingTTSManager(
            tts_callback=self._speak_chunk,
            tts_synthesize_callback=self._synthesize_chunk,
            playback_callback=self._play_streaming_response_chunk,
            synthesis_workers=synthesis_workers,
            min_chunk_chars=12,
            max_chunk_chars=150,
            max_chunk_wait_s=min(20.0, float(self.config.tts_timeout)),
        )
        self._streaming_tts_manager.start()

        self._task_status_announcer = TaskStatusAnnouncer(
            speak_func=self._speak_chunk,
            intervals=[6, 20, 45, 90, 150]
        )
        self._task_status_announcer.start(command)

    def _on_llm_complete(self, full_text: str):
        """Callback при завершении генерации LLM (WebSocket final event)."""
        self._stop_stream_watchdog()

        with self._streaming_completion_lock:
            if self._streaming_generation == 0:
                kiwi_log("LLM", "Duplicate/stale completion callback ignored", level="WARNING")
                return
            self._streaming_generation = 0  # mark as handled

        # FALLBACK: если full_text пуст, но есть накопленный текст в WebSocket клиенте
        if not full_text and hasattr(self.openclaw, '_accumulated_text'):
            accumulated = self.openclaw._accumulated_text
            if accumulated:
                kiwi_log("LLM", f"Generation complete with EMPTY final, using accumulated: {len(accumulated)} chars", level="WARNING")
                full_text = accumulated
            else:
                kiwi_log("LLM", "Generation complete: 0 chars (EMPTY)", level="WARNING")
        else:
            kiwi_log("LLM", f"Generation complete: {len(full_text)} chars", level="INFO")
        
        # Останавливаем TaskStatusAnnouncer
        if self._task_status_announcer:
            self._task_status_announcer.stop()
            self._task_status_announcer = None
        
        # StreamingTTSManager остановится сам при получении None в очередь
        if self._streaming_tts_manager:
            self._streaming_tts_manager.stop()
            self._streaming_tts_manager = None

        # Safety fallback: если в streaming-ответе ни один чанк не начал проигрываться,
        # озвучиваем final-текст обычным путём, чтобы не оставаться в тишине.
        if (not self._streaming_response_playback_started) and full_text and full_text.strip():
            kiwi_log("STREAM-TTS", "No playback started for this response, using fallback speak()", level="WARNING")
            self.speak(full_text, style=self._streaming_style)
            # speak() сам сбрасывает _is_speaking и _set_state() через
            # _play_audio_interruptible / _speak_streaming, но idle timer
            # и dialog mode нужно гарантировать здесь.
            if not self._barge_in_requested:
                self._start_idle_timer()
            return

        # Для streaming-ветки вручную делаем тот же post-playback переход,
        # что и в _play_audio_interruptible().
        self._is_speaking = False
        self.listener._tts_start_time = time.time()
        self.listener.activate_dialog_mode()

        if self.listener.dialog_mode:
            self._set_state(DialogueState.LISTENING)
        else:
            self._set_state(DialogueState.IDLE)

        if not self._barge_in_requested:
            self._start_idle_timer()
    
    def _generate_tts_audio(
        self,
        text: str,
        style: str = "neutral",
        voice: Optional[str] = None,
        language: str = "Russian",
        use_cache: bool = True,
    ):
        """Unified TTS generation — delegates to self.tts.synthesize()."""
        resolved_style = style or self.config.tts_default_style
        kwargs = {}
        if self.tts_provider == "elevenlabs":
            voice = voice or self.config.tts_elevenlabs_voice_id
            kwargs.update(
                model_id=self.config.tts_elevenlabs_model_id,
                output_format=self.config.tts_elevenlabs_output_format,
                use_streaming_endpoint=self.config.tts_elevenlabs_use_streaming_endpoint,
                optimize_streaming_latency=self.config.tts_elevenlabs_optimize_streaming_latency,
                similarity_boost=self.config.tts_elevenlabs_similarity_boost,
                use_speaker_boost=self.config.tts_elevenlabs_use_speaker_boost,
            )
        elif self.tts_provider != "piper":
            voice = voice or self.config.tts_voice
            kwargs["model_size"] = self.config.tts_model_size
        return self.tts.synthesize(
            text=text,
            voice=voice,
            style=resolved_style,
            language=language,
            use_cache=use_cache,
            **kwargs,
        )
    
    def _synthesize_chunk(self, chunk: str) -> Optional[Tuple[np.ndarray, int]]:
        """Генерирует аудио для одного чанка текста (для streaming TTS)."""
        if not chunk or not chunk.strip():
            return None

        try:
            clean_chunk = clean_chunk_for_tts(chunk)

            if not clean_chunk:
                kiwi_log("TTS-CHUNK", "Skipping empty chunk after cleaning", level="INFO")
                return None

            if "'type':" in clean_chunk or '"type":' in clean_chunk or '}{' in clean_chunk:
                kiwi_log("TTS-CHUNK", f"Chunk still contains JSON patterns, skipping: {clean_chunk[:60]}...", level="WARNING")
                return None

            kiwi_log("TTS-CHUNK", f"Synthesizing ({self.tts_provider}): {clean_chunk[:60]}...", level="INFO")
            started = time.time()

            if self.tts_provider == "elevenlabs":
                audio, sample_rate = self.tts.synthesize(
                    text=clean_chunk,
                    voice=self.config.tts_elevenlabs_voice_id,
                    style=self._streaming_style,
                    language="Russian",
                    use_cache=True,
                    model_id=self.config.tts_elevenlabs_model_id,
                    output_format=self.config.tts_elevenlabs_output_format,
                    use_streaming_endpoint=self.config.tts_elevenlabs_use_streaming_endpoint,
                    optimize_streaming_latency=self.config.tts_elevenlabs_optimize_streaming_latency,
                    stability=self.config.tts_elevenlabs_stability,
                    similarity_boost=self.config.tts_elevenlabs_similarity_boost,
                    style_value=self.config.tts_elevenlabs_style,
                    use_speaker_boost=self.config.tts_elevenlabs_use_speaker_boost,
                    speed=self.config.tts_elevenlabs_speed,
                )

                # Fallback to streaming endpoint if non-stream returned no audio.
                if (audio is None or len(audio) == 0) and self.config.tts_elevenlabs_use_streaming_endpoint:
                    kiwi_log("TTS-CHUNK", "Retry via ElevenLabs streaming endpoint...", level="INFO")
                    audio, sample_rate = self.tts.synthesize(
                        text=clean_chunk,
                        voice=self.config.tts_elevenlabs_voice_id,
                        style=self._streaming_style,
                        language="Russian",
                        use_cache=True,
                        model_id=self.config.tts_elevenlabs_model_id,
                        output_format=self.config.tts_elevenlabs_output_format,
                        use_streaming_endpoint=True,
                        optimize_streaming_latency=self.config.tts_elevenlabs_optimize_streaming_latency,
                        stability=self.config.tts_elevenlabs_stability,
                        similarity_boost=self.config.tts_elevenlabs_similarity_boost,
                        style_value=self.config.tts_elevenlabs_style,
                        use_speaker_boost=self.config.tts_elevenlabs_use_speaker_boost,
                        speed=self.config.tts_elevenlabs_speed,
                    )
            else:
                audio, sample_rate = self._generate_tts_audio(
                    text=clean_chunk,
                    style=self._streaming_style,
                    voice=None,
                    language="Russian",
                    use_cache=True,
                )

            if audio is None or len(audio) == 0:
                return None
            elapsed = time.time() - started
            kiwi_log("TTS-CHUNK", f"Synth OK in {elapsed:.2f}s", level="INFO")
            return audio, sample_rate

        except Exception as e:
            kiwi_log("TTS-CHUNK", f"Error: {e}", level="ERROR")
            return None

    def _speak_chunk(self, chunk: str):
        """Генерирует и воспроизводит один чанк текста (используется статус-озвучкой)."""
        # Don't start synthesis if response audio is already playing
        if self._is_speaking:
            return
        result = self._synthesize_chunk(chunk)
        if not result:
            return
        audio, sample_rate = result
        # Re-check: streaming response may have started during synthesis
        if self._is_speaking:
            return
        self._play_audio_chunk_streaming(audio, sample_rate)
    
    def _play_audio_chunk_streaming(self, audio: np.ndarray, sample_rate: int):
        """Воспроизводит аудио-чанк в режиме streaming (без прерывания)."""
        try:
            if audio is None or len(audio) == 0:
                kiwi_log("TTS-CHUNK", "Skip playback: empty audio", level="WARNING")
                return
            if sample_rate is None or int(sample_rate) <= 0:
                kiwi_log("TTS-CHUNK", f"Skip playback: invalid sample_rate={sample_rate}", level="WARNING")
                return

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            if peak > 1.0:
                audio = audio / peak
                peak = 1.0
            # Часто streaming-чанки приходят заметно тише стартовой фразы.
            # Подтягиваем уровень к комфортному, но ограничиваем gain.
            gain = 1.0
            if 0.0 < peak < 0.20:
                gain = min(3.0, 0.35 / peak)
                audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
                peak = float(np.max(np.abs(audio))) if audio.size else peak

            duration_s = len(audio) / float(sample_rate)
            kiwi_log(
                "TTS-CHUNK",
                f"Playing ({duration_s:.2f}s, sr={sample_rate}, "
                f"peak={peak:.3f}, gain={gain:.2f})",
                level="INFO",
            )
             
            self._is_speaking = True
            self._barge_in_requested = False
            if hasattr(self, "listener") and self.listener:
                self.listener._tts_start_time = time.time()
                self.listener._barge_in_counter = 0
            
            # Уведомляем TaskStatusAnnouncer что TTS играет
            if self._task_status_announcer:
                self._task_status_announcer.on_tts_playing(True)
             
            with self._sd_play_lock:
                sd.play(audio, sample_rate, device=self.config.output_device)

                poll_interval = 0.05
                max_duration = duration_s + 1.25  # буфер на хвост/драйвер
                started = time.time()

                interrupted = False

                # Надёжное ожидание завершения playback без reliance на sd.get_stream(),
                # которое может указывать на input stream (микрофон), а не output.
                done_event = threading.Event()

                def _wait_playback_done():
                    try:
                        sd.wait()
                    except Exception:
                        pass
                    finally:
                        done_event.set()

                wait_thread = threading.Thread(target=_wait_playback_done, daemon=True)
                wait_thread.start()

                while not done_event.wait(timeout=poll_interval):
                    if self._barge_in_requested:
                        interrupted = True
                        try:
                            sd.stop()
                        except Exception:
                            pass
                        kiwi_log("TTS-CHUNK", "Playback interrupted by barge-in", level="INFO")
                        break

                    if (time.time() - started) >= max_duration:
                        kiwi_log(
                            "TTS-CHUNK",
                            f"Playback timeout after {max_duration:.2f}s; forcing stop",
                            level="WARNING",
                        )
                        try:
                            sd.stop()
                        except Exception:
                            pass
                        break

                done_event.wait(timeout=1.0)

            elapsed = time.time() - started
            kiwi_log(
                "TTS-CHUNK",
                "Playback finished"
                + (" (interrupted)" if interrupted else "")
                + f" in {elapsed:.2f}s",
                level="INFO",
            )
             
        except Exception as e:
            kiwi_log("TTS-CHUNK", f"Playback error: {e}", level="ERROR")
        finally:
            self._is_speaking = False
            if hasattr(self, "listener") and self.listener:
                self.listener._tts_start_time = time.time()
            
            # Уведомляем TaskStatusAnnouncer что TTS закончил
            if self._task_status_announcer:
                self._task_status_announcer.on_tts_playing(False)

    def _play_streaming_response_chunk(self, audio: np.ndarray, sample_rate: int):
        """Playback callback только для чанков ответа LLM (не для status announcer)."""
        self._streaming_response_playback_started = True
        self._play_audio_chunk_streaming(audio, sample_rate)
    
    # === STATE MACHINE METHODS ===
    def _set_state(self, new_state: str):
        """Атомарная смена состояния с логированием."""
        with self._state_lock:
            old_state = self._dialogue_state
            self._dialogue_state = new_state
            self._last_state_change = time.time()
            timeout = self._state_timeouts.get(new_state)
            if timeout:
                self._state_until = time.time() + timeout
            else:
                self._state_until = float('inf')
        kiwi_log("STATE", f"{old_state} → {new_state}" + (f" (timeout: {timeout}s)" if timeout else ""), level="INFO")
        
        # Публикуем событие смены состояния
        if EVENT_BUS_AVAILABLE:
            from kiwi.event_bus import EventType
            get_event_bus().publish(
                EventType.STATE_CHANGED,
                {'old_state': old_state, 'new_state': new_state, 'timeout': timeout},
                source='kiwi_service'
            )
        
        # При переходе в PROCESSING или THINKING — drain очереди аудио
        if new_state in (DialogueState.PROCESSING, DialogueState.THINKING):
            self.listener.drain_audio_queue()
    
    def _get_state(self) -> str:
        """Получить текущее состояние."""
        with self._state_lock:
            return self._dialogue_state
    
    def _check_state_timeout(self) -> bool:
        """Проверить, не истёк ли таймаут текущего состояния."""
        with self._state_lock:
            if time.time() > self._state_until:
                return True
            return False
    
    def _is_in_state(self, *states: str) -> bool:
        """Проверить, находится ли система в одном из указанных состояний."""
        with self._state_lock:
            return self._dialogue_state in states
    
    def _load_sound_file(self, filepath: str) -> tuple:
        """Загружает MP3/WAV файл и возвращает numpy array (конвертация в моно)."""
        try:
            audio = AudioSegment.from_file(filepath)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2**15 if audio.sample_width == 2 else 2**31)
            
            # Конвертация стерео в моно (sounddevice ожидает 1D для моно, не interleaved)
            if audio.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            elif audio.channels > 2:
                samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            
            return samples, audio.frame_rate
        except Exception as e:
            kiwi_log("SOUND", f"Error loading {filepath}: {e}", level="ERROR")
            return None, 44100
    
    def _generate_beep(self) -> tuple:
        """Загружает звук подтверждения из файла."""
        sound_path = os.path.join(PROJECT_ROOT, 'sounds', 'confirmation.mp3')
        wave, sr = self._load_sound_file(sound_path)
        if wave is not None:
            kiwi_log("SOUND", f"Loaded confirmation sound: {len(wave)/sr:.2f}s", level="INFO")
            return wave, sr
        return self._generate_fallback_beep()
    
    def _generate_startup_sound(self) -> tuple:
        """Загружает звук запуска из файла."""
        sound_path = os.path.join(PROJECT_ROOT, 'sounds', 'startup.mp3')
        wave, sr = self._load_sound_file(sound_path)
        if wave is not None:
            kiwi_log("SOUND", f"Loaded startup sound: {len(wave)/sr:.2f}s", level="INFO")
            return wave, sr
        return self._generate_fallback_startup()
    
    def _generate_fallback_beep(self) -> tuple:
        """Fallback: программная генерация звука подтверждения."""
        samples = int(self._BEEP_SAMPLE_RATE * self._BEEP_DURATION)
        t = np.linspace(0, self._BEEP_DURATION, samples, dtype=np.float32)
        freq_start = self._BEEP_FREQ
        freq_end = self._BEEP_FREQ * 0.7
        freq = np.linspace(freq_start, freq_end, samples)
        wave = np.sin(2 * np.pi * freq * t)
        attack = int(samples * 0.1)
        decay = int(samples * 0.8)
        envelope = np.ones(samples, dtype=np.float32)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)
        wave *= envelope
        wave = wave / np.max(np.abs(wave)) * 0.6
        return wave.astype(np.float32), self._BEEP_SAMPLE_RATE
    
    def _generate_fallback_startup(self) -> tuple:
        """Fallback: программная генерация звука запуска."""
        duration = 0.6
        sample_rate = self._STARTUP_SAMPLE_RATE
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, dtype=np.float32)
        freqs = [261.63, 329.63, 392.00]
        wave = np.zeros(samples, dtype=np.float32)
        delays = [0.0, 0.05, 0.1]
        for i, (freq, delay) in enumerate(zip(freqs, delays)):
            delay_samples = int(delay * sample_rate)
            note_samples = samples - delay_samples
            if note_samples <= 0:
                continue
            t_note = t[delay_samples:]
            note = np.sin(2 * np.pi * freq * t_note)
            decay = int(note_samples * 0.6)
            envelope = np.ones(note_samples, dtype=np.float32)
            envelope[-decay:] = np.linspace(1, 0, decay)
            note *= envelope
            wave[delay_samples:] += note * (0.4 - i * 0.05)
        wave = wave / np.max(np.abs(wave)) * 0.7
        return wave.astype(np.float32), sample_rate
    
    def _load_idle_sound(self) -> tuple:
        """Загружает звук перехода в ожидание."""
        sound_path = os.path.join(PROJECT_ROOT, 'sounds', 'idle.mp3')
        wave, sr = self._load_sound_file(sound_path)
        if wave is not None:
            kiwi_log("SOUND", f"Loaded idle sound: {len(wave)/sr:.2f}s", level="INFO")
            return wave, sr
        wave, sr = self._generate_fallback_beep()
        wave = wave * 0.5
        return wave, sr
    
    def play_idle_sound(self):
        """Воспроизводит звук перехода в ожидание."""
        try:
            self._is_speaking = True
            kiwi_log("SOUND", "Playing idle sound...", level="INFO")
            sd.play(self._idle_sound, self._idle_sr)
            sd.wait()
            self._is_speaking = False
            # Метка завершения короткого звука (не TTS) — короткая dead zone
            self.listener._sound_end_time = time.time()
            kiwi_log("SOUND", "Idle done", level="INFO")
        except Exception as e:
            kiwi_log("SOUND", f"Error: {e}", level="ERROR")
            self._is_speaking = False
    
    def play_beep(self, async_mode=True):
        """Воспроизводит звук подтверждения команды."""
        current_time = time.time()
        if (current_time - self._last_beep_time) < 2.0:
            kiwi_log("SOUND", "Skipping beep (too soon)", level="INFO")
            return
        
        self._last_beep_time = current_time
        def _play_with_end_marker():
            try:
                sd.play(self._beep_sound, self._beep_sr)
                sd.wait()
                # Метка завершения короткого звука
                self.listener._sound_end_time = time.time()
                kiwi_log("SOUND", "Confirmation done", level="INFO")
            except Exception as e:
                kiwi_log("SOUND", f"Error: {e}", level="ERROR")

        try:
            kiwi_log("SOUND", "Playing confirmation sound...", level="INFO")
            if not async_mode:
                _play_with_end_marker()
            else:
                # Асинхронно — запускаем в потоке чтобы не блокировать
                threading.Thread(target=_play_with_end_marker, daemon=True).start()
                kiwi_log("SOUND", "Confirmation playing (async)", level="INFO")
        except Exception as e:
            kiwi_log("SOUND", f"Error: {e}", level="ERROR")
    
    def play_startup_sound(self):
        """Воспроизводит звук запуска."""
        try:
            kiwi_log("SOUND", "Playing startup sound...", level="INFO")
            sd.play(self._startup_sound, self._startup_sr)
            sd.wait()
            # Метка завершения короткого звука
            if hasattr(self, 'listener') and self.listener:
                self.listener._sound_end_time = time.time()
            kiwi_log("SOUND", "Startup done", level="INFO")
        except Exception as e:
            kiwi_log("SOUND", f"Error: {e}", level="ERROR")
    
    def start(self):
        """Запускает сервис с приветственным звуком и сообщением."""
        if self.is_running:
            return
        
        with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'a', encoding='utf-8') as f:
            f.write('[START] KiwiServiceOpenClaw.start() called\n')
        
        # Запускаем Event Bus
        if EVENT_BUS_AVAILABLE:
            from kiwi.event_bus import EventType
            get_event_bus().start()
            get_event_bus().publish(
                EventType.SYSTEM_STARTUP,
                {'version': '2.0', 'mode': 'openclaw'},
                source='kiwi_service'
            )
        
        self.play_startup_sound()
        
        self.is_running = True
        kiwi_log("KIWI", "Starting listener...", level="INFO")
        self.listener.start()
        kiwi_log("KIWI", "Kiwi Voice Service with OpenClaw started!", level="INFO")
        
        def greeting():
            time.sleep(0.5)
            self.speak(
                "Привет! Я Киви. Скажи 'Киви' и команду, когда будешь готов.",
                style="cheerful",
                allow_barge_in=False,
            )
        threading.Thread(target=greeting, daemon=True).start()
        
        with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'a', encoding='utf-8') as f:
            f.write('[START] Service fully started!\n')
    
    def stop(self):
        """Останавливает сервис."""
        # Публикуем событие остановки
        if EVENT_BUS_AVAILABLE:
            from kiwi.event_bus import EventType
            get_event_bus().publish(
                EventType.SYSTEM_SHUTDOWN,
                {},
                source='kiwi_service'
            )
            get_event_bus().stop()
        
        self.is_running = False
        self.listener.stop()
        kiwi_log("KIWI", "Kiwi Voice Service stopped", level="INFO")

    def _get_current_speaker_meta(self) -> Dict[str, Any]:
        """Получает метаданные последней реплики от listener."""
        default = {
            "speaker_id": "unknown",
            "speaker_name": "Незнакомец",
            "priority": 2,
            "confidence": 0.0,
            "music_probability": 0.0,
            "text": "",
            "timestamp": 0.0,
        }
        if hasattr(self.listener, "get_last_speaker_meta"):
            try:
                meta = self.listener.get_last_speaker_meta()
                if isinstance(meta, dict):
                    return {**default, **meta}
            except Exception as e:
                kiwi_log("SPEAKER", f"Failed to read speaker meta: {e}", level="ERROR")
        return default

    def _is_owner_speaker(self, speaker_meta: Dict[str, Any]) -> bool:
        return str(speaker_meta.get("speaker_id", "")).lower() == str(self._owner_id).lower()

    def _owner_profile_registered(self) -> bool:
        manager = getattr(self.listener, "speaker_manager", None)
        if manager is None:
            return False
        try:
            base = getattr(manager, "base_identifier", None)
            if base is not None and hasattr(base, "profiles"):
                return self._owner_id in base.profiles
        except Exception:
            return False
        return False

    def _is_small_talk_or_safe_request(self, command_lower: str) -> bool:
        """Запросы, которые не требуют отдельного owner approval."""
        safe_markers = [
            "анекдот", "шутк", "сказк", "истори", "поговори", "как дела", "спой",
            "который час", "сколько времени", "время", "дата", "день недели",
            "погода", "новости", "расскажи", "объясни", "кто такой", "что такое",
        ]
        return any(marker in command_lower for marker in safe_markers)

    def _looks_like_actionable_task(self, command_lower: str) -> bool:
        task_markers = [
            "сделай", "выполни", "запусти", "открой", "закрой", "удали", "создай",
            "поставь", "скачай", "установи", "отправь", "напиши", "измени", "перемести",
            "найди", "поищи", "проверь", "зайди", "команд", "в терминале", "shell",
        ]
        return any(marker in command_lower for marker in task_markers)

    def _approval_yes(self, command_lower: str) -> bool:
        yes_markers = ["разрешаю", "подтверждаю", "одобряю", "выполняй", "да, выполняй", "да выполняй"]
        return any(marker in command_lower for marker in yes_markers)

    def _approval_no(self, command_lower: str) -> bool:
        no_markers = ["запрещаю", "отклоняю", "не выполняй", "не надо", "нельзя", "отмена разрешения"]
        return any(marker in command_lower for marker in no_markers)

    def _extract_name_from_voice_command(self, command: str) -> Optional[str]:
        match = re.search(r"(?:запомни меня как|это мой друг|друг)\s+([а-яa-zё\-]{2,30})", command, re.IGNORECASE)
        if not match:
            return None
        name = match.group(1).strip()
        return name[:1].upper() + name[1:]
    
    def _on_wake_word(self, command: str):
        """
        Wake word handler — pipeline orchestrator.
        Each stage receives a CommandContext and may set ctx.abort = True to stop.
        """
        ctx = CommandContext(command=command)
        for stage in [
            self._stage_init_and_dedup,
            self._stage_resolve_speaker,
            self._stage_check_approval,
            self._stage_handle_special_commands,
            self._stage_handle_stop_cancel,
            self._stage_completeness_check,
            self._stage_owner_approval_gate,
            self._stage_dispatch_to_llm,
        ]:
            stage(ctx)
            if ctx.abort:
                return

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _stage_init_and_dedup(self, ctx: CommandContext) -> None:
        """Set PROCESSING, reject duplicates, extend dialog timeout."""
        self._set_state(DialogueState.PROCESSING)

        kiwi_log("KIWI", f"Услышала: {ctx.command}", level="INFO")

        ctx.timestamp = time.time()
        if ctx.command == self._last_command and (ctx.timestamp - self._last_command_time) < self._command_cooldown:
            kiwi_log("DEDUP", f"Ignoring duplicate command within {self._command_cooldown}s", level="INFO")
            self._set_state(DialogueState.IDLE)
            ctx.abort = True
            return

        self._last_command = ctx.command
        self._last_command_time = ctx.timestamp

        # Продлеваем dialog mode на время обработки
        if self.listener.dialog_mode:
            self.listener.dialog_until = time.time() + self.listener.dialog_timeout
            kiwi_log("DIALOG", "Extended timeout for processing", level="INFO")

        ctx.command_lower = ctx.command.lower().strip()

    def _stage_resolve_speaker(self, ctx: CommandContext) -> None:
        """Get speaker meta, log, show owner warning once."""
        speaker_meta = self._get_current_speaker_meta()
        ctx.speaker_id = str(speaker_meta.get("speaker_id", "unknown"))
        ctx.speaker_name = str(speaker_meta.get("speaker_name", "Незнакомец"))
        ctx.speaker_confidence = float(speaker_meta.get("confidence", 0.0))
        ctx.speaker_music_prob = float(speaker_meta.get("music_probability", 0.0))
        ctx.is_owner = self._is_owner_speaker(speaker_meta)
        ctx.owner_profile_ready = self._owner_profile_registered()

        kiwi_log(
            "SPEAKER",
            f"command from {ctx.speaker_name} ({ctx.speaker_id}), "
            f"owner={ctx.is_owner}, conf={ctx.speaker_confidence:.2f}, music={ctx.speaker_music_prob:.2f}",
            level="INFO",
        )

        if not ctx.owner_profile_ready and not self._owner_profile_warning_shown:
            kiwi_log(
                "APPROVAL",
                f"Owner profile '{self._owner_id}' is not registered yet. "
                f"Using name-based fallback approval; voice enrollment is recommended.",
                level="WARNING",
            )
            self._owner_profile_warning_shown = True

    def _stage_check_approval(self, ctx: CommandContext) -> None:
        """Expire stale approvals, handle owner yes/no."""
        # Очищаем устаревшее pending-одобрение
        if self._pending_owner_approval:
            age = time.time() - float(self._pending_owner_approval.get("timestamp", 0.0))
            if age > self._owner_approval_timeout:
                kiwi_log("APPROVAL", f"Pending request expired ({age:.1f}s)", level="INFO")
                self._pending_owner_approval = None

        # Если есть pending задача и говорит owner — ждём yes/no подтверждение
        fallback_owner_phrase = (not ctx.owner_profile_ready) and (self._owner_name.lower() in ctx.command_lower)
        if self._pending_owner_approval and (ctx.is_owner or fallback_owner_phrase):
            if self._approval_yes(ctx.command_lower):
                ctx.approved_command_from_owner = str(self._pending_owner_approval.get("command", "")).strip()
                requester = str(self._pending_owner_approval.get("speaker_name", "кто-то"))
                self._pending_owner_approval = None
                if ctx.approved_command_from_owner:
                    kiwi_log("APPROVAL", f"Approved by owner. Running deferred command from {requester}", level="INFO")
                    self.speak(f"Принято. Выполняю запрос от {requester}.", style="confident")
                    ctx.command = ctx.approved_command_from_owner
                    ctx.command_lower = ctx.command.lower().strip()
            elif self._approval_no(ctx.command_lower):
                requester = str(self._pending_owner_approval.get("speaker_name", "кто-то"))
                self._pending_owner_approval = None
                kiwi_log("APPROVAL", f"Denied by owner. Requester={requester}", level="INFO")
                self.speak(f"Хорошо, запрос от {requester} отклонён.", style="calm")
                ctx.abort = True

    def _stage_handle_special_commands(self, ctx: CommandContext) -> None:
        """Reset context, calibrate, voice profile commands."""
        calibrate_words = ['калибровка', 'калибруй', 'перекалибруй', 'обнови профиль']
        reset_prompt_words = ['сбрось контекст', 'новый разговор', 'забудь', 'сбрось системный промпт']

        if any(word in ctx.command_lower for word in reset_prompt_words):
            kiwi_log("KIWI", "Resetting system prompt...", level="INFO")
            self._system_prompt_sent = False
            self.speak("Контекст сброшен. Начинаем новый разговор.", style="neutral")
            ctx.abort = True
            return

        if any(word in ctx.command_lower for word in calibrate_words):
            kiwi_log("KIWI", "Speaker ID calibration requested", level="INFO")
            self._self_profile_created = False
            self.speak("Калибрую распознавание голоса.", style="neutral")
            ctx.abort = True
            return

        # === КОМАНДЫ УПРАВЛЕНИЯ ГОЛОСОВЫМИ ПРОФИЛЯМИ ===
        owner_register_words = [
            "запомни мой голос",
            "зарегистрируй мой голос",
            "я хозяин",
        ]
        # Dynamic: "это <owner_name>" / "я <owner_name>"
        _on = self._owner_name.lower()
        if _on and _on != "owner":
            owner_register_words.extend([f"это {_on}", f"я {_on}"])
        if any(word in ctx.command_lower for word in owner_register_words):
            if hasattr(self.listener, "register_owner_voice"):
                success = self.listener.register_owner_voice(self._owner_name)
                if success:
                    self.speak(f"Готово. Я запомнила голос владельца: {self._owner_name}.", style="cheerful")
                else:
                    self.speak("Не получилось сохранить профиль. Скажи фразу подлиннее и попробуй снова.", style="calm")
            else:
                self.speak("Модуль профилей сейчас недоступен.", style="calm")
            ctx.abort = True
            return

        if "кто говорит" in ctx.command_lower or "кто это говорит" in ctx.command_lower:
            if hasattr(self.listener, "describe_last_speaker"):
                self.speak(self.listener.describe_last_speaker(), style="neutral")
            else:
                self.speak("Сейчас не могу определить говорящего.", style="calm")
            ctx.abort = True
            return

        if "какие голоса" in ctx.command_lower or "список голосов" in ctx.command_lower:
            if hasattr(self.listener, "describe_known_voices"):
                self.speak(self.listener.describe_known_voices(), style="neutral")
            else:
                self.speak("Список голосов недоступен.", style="calm")
            ctx.abort = True
            return

        if "запомни меня как" in ctx.command_lower or "это мой друг" in ctx.command_lower:
            name = self._extract_name_from_voice_command(ctx.command)
            if not name:
                self.speak("Скажи имя после фразы: запомни меня как ...", style="neutral")
                ctx.abort = True
                return
            if hasattr(self.listener, "remember_last_voice_as"):
                success, _sid = self.listener.remember_last_voice_as(name)
                if success:
                    self.speak(f"Запомнила голос: {name}.", style="cheerful")
                else:
                    self.speak("Не удалось сохранить голос. Нужна более чёткая фраза.", style="calm")
            else:
                self.speak("Модуль профилей сейчас недоступен.", style="calm")
            ctx.abort = True
            return

    def _stage_handle_stop_cancel(self, ctx: CommandContext) -> None:
        """Stop TTS + barge-in, cancel OpenClaw."""
        stop_words = ['стоп', 'отмена', 'хватит', 'прекрати', 'остановись', 'стой', 'cancel', 'stop']
        if not any(word in ctx.command_lower for word in stop_words):
            return

        kiwi_log("KIWI", "Получена команда отмены!", level="INFO")

        tts_was_active = self.is_speaking() or self._is_streaming or self._streaming_tts_manager is not None
        openclaw_was_processing = self.openclaw.is_processing()

        if tts_was_active:
            self.request_barge_in()
            self._streaming_stop_event.set()
            self._stop_stream_watchdog()
            if self._streaming_tts_manager:
                self._streaming_tts_manager.stop(graceful=False)
                self._streaming_tts_manager = None

        cancelled = self.openclaw.cancel() if openclaw_was_processing else False

        if tts_was_active:
            # Пользователь попросил "стой" во время речи:
            # молча прерываем и возвращаемся в режим ожидания новой команды.
            self._is_streaming = False
            self.listener.activate_dialog_mode()
            self._set_state(DialogueState.LISTENING)
            self._start_idle_timer()
            ctx.abort = True
            return

        if cancelled:
            self.speak("Остановила.", style="calm")
        else:
            self.speak("Нечего останавливать.", style="neutral")
        ctx.abort = True

    def _stage_completeness_check(self, ctx: CommandContext) -> None:
        """Combine pending phrase, check completeness."""
        combined_text = ctx.command
        if self._pending_phrase:
            elapsed = time.time() - self._pending_timestamp
            if elapsed < self._pending_timeout:
                combined_text = f"{self._pending_phrase} {ctx.command}".strip()
                kiwi_log("KIWI", f"Combined with pending: '{self._pending_phrase}' + '{ctx.command}' → '{combined_text}'", level="INFO")
            else:
                kiwi_log("KIWI", f"Pending phrase expired ({elapsed:.1f}s)", level="INFO")
                self._pending_phrase = ""

        # Быстрая локальная проверка completeness
        quick_complete = self._quick_completeness_check(combined_text)
        if quick_complete:
            kiwi_log("COMPLETE-CHECK", "Quick check: COMPLETE", level="INFO")
        else:
            kiwi_log("COMPLETE-CHECK", "Quick check: INCOMPLETE - waiting for more", level="INFO")
            # Фраза не завершена - сохраняем и ждем продолжения
            self._pending_phrase = combined_text
            self._pending_timestamp = time.time()
            if self.listener.dialog_mode:
                self.listener.dialog_until = time.time() + self._pending_timeout
                kiwi_log("DIALOG", f"Extended timeout for pending phrase ({self._pending_timeout}s)", level="INFO")
            ctx.abort = True
            return

        # Фраза complete - сбрасываем pending и отправляем в OpenClaw
        self._pending_phrase = ""
        ctx.command = combined_text
        ctx.command_lower = ctx.command.lower().strip()

    def _stage_owner_approval_gate(self, ctx: CommandContext) -> None:
        """Defer non-owner actionable tasks."""
        if (
            not ctx.approved_command_from_owner
            and not ctx.is_owner
            and self._looks_like_actionable_task(ctx.command_lower)
            and not self._is_small_talk_or_safe_request(ctx.command_lower)
        ):
            self._pending_owner_approval = {
                "command": ctx.command,
                "speaker_id": ctx.speaker_id,
                "speaker_name": ctx.speaker_name,
                "timestamp": time.time(),
            }
            kiwi_log("APPROVAL", f"Pending owner approval for speaker={ctx.speaker_name}, command='{ctx.command}'", level="INFO")
            approve_hint = "Скажи: разрешаю или запрещаю."
            if not ctx.owner_profile_ready:
                approve_hint = f"Скажи: {self._owner_name}, разрешаю. Или: {self._owner_name}, запрещаю."
            self.speak(
                f"{self._owner_name}, {ctx.speaker_name} просит: {ctx.command}. "
                f"{approve_hint}",
                style="neutral",
            )
            self.listener.activate_dialog_mode()
            self._set_state(DialogueState.LISTENING)
            ctx.abort = True

    def _stage_dispatch_to_llm(self, ctx: CommandContext) -> None:
        """Event, beep, THINKING, streaming/blocking dispatch."""
        # Публикуем событие получения команды
        if EVENT_BUS_AVAILABLE:
            from kiwi.event_bus import EventType
            get_event_bus().publish(
                EventType.COMMAND_RECEIVED,
                {'command': ctx.command, 'source': 'voice'},
                source='kiwi_service'
            )

        # Запускаем beep асинхронно
        self.play_beep(async_mode=True)

        # === ОБРАБОТКА КОМАНДЫ ===
        kiwi_log("KIWI", "Думаю...", level="INFO")

        # === STATE MACHINE: Переход в THINKING ===
        self._set_state(DialogueState.THINKING)

        # Продлеваем dialog mode на время ожидания ответа от LLM
        if self.listener.dialog_mode:
            self.listener.dialog_until = time.time() + self.config.openclaw_timeout + 10
            kiwi_log("DIALOG", f"Extended timeout for LLM response ({self.config.openclaw_timeout + 10}s)", level="INFO")

        # Определяем эмоцию заранее (до начала стриминга)
        self._streaming_style = self._detect_emotion(ctx.command, "")

        # === STREAMING FLOW (WebSocket) или CLASSIC FLOW (blocking final response) ===
        local_qwen_no_streaming = (
            self.tts_provider == "qwen3" and self.tts_qwen_backend == "local"
        )
        use_streaming_flow = (
            self._use_websocket
            and hasattr(self.openclaw, 'send_message')
            and not local_qwen_no_streaming
        )

        if use_streaming_flow:
            self._dispatch_streaming(ctx)
        else:
            self._dispatch_blocking(ctx, local_qwen_no_streaming)

    # ------------------------------------------------------------------
    # Dispatch helpers (called from _stage_dispatch_to_llm)
    # ------------------------------------------------------------------

    def _dispatch_streaming(self, ctx: CommandContext) -> None:
        """Streaming: launch TTS manager and send message via WebSocket."""
        kiwi_log("KIWI", "Using streaming LLM + TTS", level="INFO")
        with self._streaming_completion_lock:
            self._streaming_generation += 1
        self._streaming_response_playback_started = False

        # Отменяем предыдущий запрос к OpenClaw если он ещё выполняется
        if self.openclaw.is_processing():
            kiwi_log("KIWI", "Cancelling previous OpenClaw request", level="INFO")
            self.openclaw.cancel()
        self._start_streaming_runtime(ctx.command)

        # Отправляем сообщение с системным промтом как контекст ТОЛЬКО при первом запросе
        # Это "задаёт" системный промт для сессии
        if not self._system_prompt_sent:
            kiwi_log("KIWI", "Sending first message with system prompt", level="INFO")
            success = self.openclaw.send_message(
                ctx.command,
                context=self.config.voice_system_prompt
            )
            self._system_prompt_sent = True
        else:
            success = self.openclaw.send_message(ctx.command)

        if not success:
            kiwi_log("KIWI", "Failed to send message via WebSocket", level="ERROR")
            self._stop_stream_watchdog()
            self._streaming_tts_manager.stop(graceful=False)
            self._streaming_tts_manager = None
            if self._task_status_announcer:
                self._task_status_announcer.stop()
                self._task_status_announcer = None
            self.speak("Извини, не удалось отправить сообщение.", style="calm")
            self._set_state(DialogueState.IDLE)
        else:
            self._start_stream_watchdog(ctx.command)

    def _dispatch_blocking(self, ctx: CommandContext, local_qwen_no_streaming: bool) -> None:
        """Blocking flow: CLI or WebSocket without streaming TTS."""
        self._stop_stream_watchdog()
        if self._use_websocket and hasattr(self.openclaw, "chat"):
            if local_qwen_no_streaming:
                kiwi_log("KIWI", "Streaming TTS disabled for local Qwen; using blocking final response", level="INFO")
            else:
                kiwi_log("KIWI", "Using blocking flow (WebSocket mode)", level="INFO")
        else:
            kiwi_log("KIWI", "Using classic blocking flow (CLI mode)", level="INFO")

        # Для blocking-flow добавляем системный промт к первому сообщению
        if not self._system_prompt_sent:
            full_command = f"{self.config.voice_system_prompt}\n\n{ctx.command}"
            self._system_prompt_sent = True
            kiwi_log("KIWI", "System prompt added to first message", level="INFO")
        else:
            full_command = ctx.command

        response = self.openclaw.chat(full_command)

        # Нормализация ответа
        if isinstance(response, list):
            response = "".join(str(r) for r in response)

        # Если ответ [SILENCE] - не говорим вслух
        if response and response.strip().upper() == "[SILENCE]":
            kiwi_log("KIWI", "Silenced response (likely noise/incomplete)", level="INFO")
            self.listener.dialog_mode = False
            self._set_state(DialogueState.IDLE)
            return

        if response:
            kiwi_log("KIWI", f"Ответ: {response[:100]}...", level="INFO")
            style = self._detect_emotion(ctx.command, response)
            self.speak(response, style=style)
    
    def _quick_completeness_check(self, text: str) -> bool:
        """
        Быстрая локальная проверка completeness без LLM.
        Returns True если фраза явно complete, False если нужен LLM.
        """
        stripped = text.strip().lower()
        words = stripped.split()
        
        if len(stripped) < 5:
            return False
        
        # Заканчивается на знак препинания — явно complete
        if stripped.endswith(('.', '!', '?')):
            return True
        
        # Длинная фраза без незавершённых паттернов — скорее всего complete
        incomplete_endings = {
            'и', 'а', 'но', 'или', 'да', 'либо', 'тоже', 'также',
            'что', 'чтобы', 'когда', 'если', 'хотя', 'потому', 'так', 
            'который', 'которая', 'которое', 'которые',
            'какой', 'какая', 'какое', 'какие',
            'кто', 'чей', 'где', 'куда', 'откуда',
            'в', 'на', 'с', 'под', 'над', 'за', 'перед', 'при',
            'к', 'по', 'у', 'о', 'об', 'до', 'от', 'для', 'без',
        }
        incomplete_patterns = [
            'я хочу', 'я буду', 'я собираюсь', 'мне нужно', 'надо бы',
            'давай', 'скажи', 'расскажи', 'покажи', 'объясни', 'помоги',
        ]
        
        # Если >= 5 слов и не заканчивается на союз/предлог
        if len(words) >= 5:
            last_word = words[-1].rstrip('.,!?')
            if last_word not in incomplete_endings and not stripped.endswith((',', '...')):
                # Проверяем на незавершённые паттерны
                for pattern in incomplete_patterns:
                    if stripped.endswith(pattern):
                        return False
                return True
        
        # Явно завершённые фразы
        complete_patterns = [
            'что-нибудь', 'что-то', 'всё', 'ничего', 'пожалуйста',
            'анекдот', 'историю', 'сказку', 'шутку', 'время', 'дату', 'погоду',
        ]
        for pattern in complete_patterns:
            if pattern in stripped:
                return True
        
        return False
    
    
    def _detect_emotion(self, command: str, response: str) -> str:
        """Определяет эмоциональный стиль для ответа."""
        command_lower = command.lower()
        response_lower = response.lower()
        
        if any(w in command_lower for w in ["срочно", "быстро", "важно"]):
            return "confident"
        if any(w in command_lower for w in ["грустно", "плохо", "ужасно", "грустный"]):
            return "sad"
        if any(w in command_lower for w in ["круто", "отлично", "супер", "ура", "радостно"]):
            return "excited"
        if any(w in command_lower for w in ["тихо", "секрет", "шёпотом"]):
            return "whisper"
        if any(w in command_lower for w in ["пошути", "анекдот", "смешно"]):
            return "playful"
        if any(w in command_lower for w in ["расскажи", "объясни", "что такое"]):
            return "neutral"
        
        if any(w in response_lower for w in ["извини", "к сожалению", "не могу"]):
            return "calm"
        if any(w in response_lower for w in ["!", "отлично", "здорово", "супер"]):
            return "cheerful"
        
        if "?" in command:
            return "playful"
        
        return "neutral"
    
    def _speak_streaming(self, text: str, style: str = "neutral", voice: Optional[str] = None, language: str = "Russian"):
        """Стриминговое воспроизведение: генерирует и воспроизводит чанки параллельно."""
        # Разбиваем текст на чанки
        chunks = split_text_into_chunks(text, max_chunk_size=150)
        kiwi_log("TTS-STREAM", f"Text split into {len(chunks)} chunks", level="INFO")

        # Очищаем очередь и запускаем playback поток
        self._clear_audio_queue()
        self._streaming_stop_event.clear()
        self._barge_in_requested = False
        self._is_streaming = True
        
        # Запускаем поток воспроизведения
        playback_thread = threading.Thread(
            target=self._streaming_playback_loop,
            args=(len(chunks),),
            daemon=True
        )
        playback_thread.start()
        
        # Генерируем чанки и кладём в очередь
        for i, chunk in enumerate(chunks):
            if self._streaming_stop_event.is_set():
                kiwi_log("TTS-STREAM", "Streaming cancelled", level="INFO")
                break
            
            try:
                kiwi_log("TTS-STREAM", f"Generating chunk {i+1}/{len(chunks)}: {chunk[:50]}...", level="INFO")
                
                audio, sample_rate = self._generate_tts_audio(
                    text=chunk,
                    style=style,
                    voice=voice,
                    language=language,
                    use_cache=True,
                )
                
                if audio is not None:
                    # Кладём в очередь: (audio, sample_rate, chunk_index)
                    self._audio_queue.put((audio, sample_rate, i))
                    kiwi_log("TTS-STREAM", f"Chunk {i+1} ready ({len(audio)/sample_rate:.2f}s)", level="INFO")
                else:
                    kiwi_log("TTS-STREAM", f"Chunk {i+1} failed", level="WARNING")
                    
            except Exception as e:
                kiwi_log("TTS-STREAM", f"Error generating chunk {i+1}: {e}", level="ERROR")
        
        # Сигнализируем об окончании генерации
        self._audio_queue.put(None)
        
        # Ждём завершения воспроизведения
        playback_thread.join(timeout=30)
        self._is_streaming = False
        self._is_speaking = False

        # Обновляем TTS timestamp для post-TTS dead zone
        self.listener._tts_start_time = time.time()

        # Активируем dialog mode ДО проверки — чтобы переход в LISTENING был корректным
        self.listener.activate_dialog_mode()

        # Сброс состояния (аналогично _play_audio_interruptible)
        if self.listener.dialog_mode:
            self._set_state(DialogueState.LISTENING)
        else:
            self._set_state(DialogueState.IDLE)

        if not self._barge_in_requested:
            self._start_idle_timer()

        kiwi_log("TTS-STREAM", "Streaming finished", level="INFO")
    
    def _streaming_playback_loop(self, total_chunks: int):
        """Поток воспроизведения стриминговых чанков."""
        chunks_played = 0
        
        while not self._streaming_stop_event.is_set():
            try:
                # Ждём данные из очереди (таймаут 0.1s)
                item = self._audio_queue.get(timeout=0.1)
                
                # None = конец стрима
                if item is None:
                    break
                
                audio, sample_rate, chunk_index = item
                
                # Проверяем barge-in
                if self._barge_in_requested:
                    kiwi_log("TTS-STREAM", "Barge-in detected, stopping stream", level="INFO")
                    break
                
                # Воспроизводим чанк
                if audio is not None and len(audio) > 0:
                    kiwi_log("TTS-STREAM", f"Playing chunk {chunk_index+1}/{total_chunks}", level="INFO")
                    self._play_audio_chunk(audio, sample_rate)
                    chunks_played += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                kiwi_log("TTS-STREAM", f"Playback error: {e}", level="ERROR")

        kiwi_log("TTS-STREAM", f"Played {chunks_played}/{total_chunks} chunks", level="INFO")
    
    def _play_audio_chunk(self, audio: np.ndarray, sample_rate: int):
        """Воспроизводит один аудио-чанк."""
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Сигнализируем listener, что Kiwi сейчас озвучивает чанк.
            self._is_speaking = True
            if hasattr(self, "listener") and self.listener:
                self.listener._tts_start_time = time.time()
                self.listener._barge_in_counter = 0
            if self._task_status_announcer:
                self._task_status_announcer.on_tts_playing(True)

            sd.play(audio, sample_rate, device=self.config.output_device)
            output_stream = sd.get_stream()

            # Ждём завершения с проверкой barge-in
            poll_interval = 0.05
            start_time = time.time()
            max_duration = len(audio) / sample_rate + 0.5  # +0.5s buffer

            while time.time() - start_time < max_duration:
                if self._barge_in_requested:
                    if output_stream is not None:
                        output_stream.stop()
                    break
                if output_stream is not None and not output_stream.active:
                    break
                time.sleep(poll_interval)
                
        except Exception as e:
            kiwi_log("TTS-STREAM", f"Chunk play error: {e}", level="ERROR")
        finally:
            self._is_speaking = False
            # Обновляем момент окончания TTS для post-TTS dead zone в listener.
            if hasattr(self, "listener") and self.listener:
                self.listener._tts_start_time = time.time()
            if self._task_status_announcer:
                self._task_status_announcer.on_tts_playing(False)
    
    def _clear_audio_queue(self):
        """Очищает очередь аудио."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def speak(
        self,
        text: str,
        style: str = "neutral",
        voice: Optional[str] = None,
        language: str = "Russian",
        allow_barge_in: bool = True,
    ):
        """Генерирует речь и воспроизводит через колонки (с поддержкой стриминга для длинных текстов)."""
        if not text or not text.strip():
            return
        
        # === Защита: парсинг строкового представления dict (если просочилось из OpenClaw) ===
        if isinstance(text, str) and text.startswith('{') and "'text'" in text:
            try:
                import ast
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict) and 'text' in parsed:
                    kiwi_log("SPEAK", f"Extracted text from dict string: {parsed['text'][:50]}...", level="INFO")
                    text = parsed['text']
            except (ValueError, SyntaxError, TypeError):
                pass
        
        # === STATE MACHINE: Переход в SPEAKING ===
        self._set_state(DialogueState.SPEAKING)
        
        # Убираем эмодзи — TTS их читает как текст
        text = normalize_tts_text(text)
        
        if not text:
            # === STATE MACHINE: Возврат в IDLE если нет текста ===
            self._set_state(DialogueState.IDLE)
            return
        
        # Для длинных текстов (>200 символов) используем стриминг
        if len(text) > 200:
            kiwi_log("TTS", f"Using streaming mode for {len(text)} chars", level="INFO")
            self._speak_streaming(text, style, voice, language)
        else:
            # Короткий текст - классический режим
            max_len = 500
            if len(text) > max_len:
                text = text[:max_len].rsplit('.', 1)[0] + '.'
                kiwi_log("TTS", f"Text truncated to {len(text)} chars", level="INFO")
            
            try:
                if self.tts_provider == "piper":
                    kiwi_log("TTS", f"Piper: '{text[:60]}...'", level="INFO")
                elif self.tts_provider == "elevenlabs":
                    kiwi_log(
                        "TTS",
                        f"ElevenLabs ({self.config.tts_elevenlabs_model_id}, "
                        f"voice={self.config.tts_elevenlabs_voice_id}): '{text[:60]}...' style={style}",
                        level="INFO",
                    )
                elif self.tts_provider == "qwen3" and self.tts_qwen_backend == "local":
                    kiwi_log("TTS", f"Qwen local {self.config.tts_model_size}: '{text[:60]}...' style={style}", level="INFO")
                else:
                    kiwi_log("TTS", f"Qwen RunPod {self.config.tts_model_size}: '{text[:60]}...' style={style}", level="INFO")

                audio, sample_rate = self._generate_tts_audio(
                    text=text.strip(),
                    style=style,
                    voice=voice,
                    language=language,
                    use_cache=True,
                )
                
                if audio is None:
                    kiwi_log("ERR", "TTS generation failed", level="ERROR")
                    self._set_state(DialogueState.IDLE)
                    return
                
                kiwi_log("TTS", f"Audio generated: {len(audio)/sample_rate:.2f}s", level="INFO")
                
                if not self._self_profile_created:
                    try:
                        success = self.listener.create_self_profile(audio, sample_rate)
                        if success:
                            self._self_profile_created = True
                            kiwi_log("SPEAKER", "Self-profile created from first TTS", level="INFO")
                    except Exception as e:
                        kiwi_log("SPEAKER", f"Failed to create self-profile: {e}", level="ERROR")
                
                self._play_audio_interruptible(audio, sample_rate, allow_barge_in=allow_barge_in)
                
            except Exception as e:
                kiwi_log("ERR", f"Speak error: {e}", level="ERROR")
                import traceback
                traceback.print_exc()
                self._set_state(DialogueState.IDLE)
        
        self.listener.activate_dialog_mode()
    
    def _play_audio(self, audio: np.ndarray, sample_rate: int):
        """Воспроизводит аудио через колонки."""
        self._play_audio_interruptible(audio, sample_rate)

    def _play_audio_interruptible(
        self,
        audio: np.ndarray,
        sample_rate: int,
        allow_barge_in: bool = True,
    ):
        """Воспроизводит аудио с возможностью прерывания голосом."""
        try:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            self._is_speaking = True
            self._barge_in_requested = False
            
            self.listener._tts_start_time = time.time()
            self.listener._barge_in_counter = 0
            
            interrupted_by_barge_in = False
            kiwi_log("TTS", f"Starting playback ({len(audio)/sample_rate:.2f}s) with smart barge-in...", level="INFO")
            sd.play(audio, sample_rate, device=self.config.output_device)

            poll_interval = 0.05
            try:
                import sounddevice as sd_module
                output_stream = sd_module.get_stream()
                if output_stream is None:
                    sd.wait()
                else:
                    while output_stream.active:
                        if allow_barge_in and self._barge_in_requested:
                            interrupted_by_barge_in = True
                            kiwi_log("BARGE-IN", "Stopping TTS playback", level="INFO")
                            output_stream.stop()
                            break
                        time.sleep(poll_interval)
            except RuntimeError:
                sd.wait()
            
            kiwi_log("TTS", "Playback finished" + (" (interrupted)" if interrupted_by_barge_in else ""), level="INFO")
            
            self._is_speaking = False
            self.listener._tts_start_time = time.time()
            
            # === STATE MACHINE: Возврат в IDLE или LISTENING ===
            if self.listener.dialog_mode:
                self._set_state(DialogueState.LISTENING)
            else:
                self._set_state(DialogueState.IDLE)
            
            if not interrupted_by_barge_in:
                self._start_idle_timer()
            
        except Exception as e:
            kiwi_log("ERR", f"Playback error: {e}", level="ERROR")
            self._is_speaking = False
            # === STATE MACHINE: Возврат в IDLE при ошибке ===
            self._set_state(DialogueState.IDLE)
    
    def _start_idle_timer(self):
        """Запускает таймер для idle звука через 1.5 секунды."""
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None
        
        kiwi_log("IDLE", f"Starting idle timer ({self._idle_delay}s)...", level="INFO")
        
        def _play_idle_after_delay():
            if self._is_speaking:
                kiwi_log("IDLE", "Skipping - Kiwi is speaking", level="INFO")
                return
            if not self.listener.dialog_mode:
                kiwi_log("IDLE", "Skipping - not in dialog mode", level="INFO")
                return

            kiwi_log("IDLE", "Timer expired, playing idle sound", level="INFO")
            self.play_idle_sound()
        
        self._idle_timer = threading.Timer(self._idle_delay, _play_idle_after_delay)
        self._idle_timer.daemon = True
        self._idle_timer.start()
    
    def _cancel_idle_timer(self):
        """Отменяет таймер idle звука."""
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None
            kiwi_log("IDLE", "Timer cancelled", level="INFO")
    
    def request_barge_in(self):
        """Вызывается listener когда обнаружен голос пользователя во время TTS."""
        if self._is_speaking or self._is_streaming or self._streaming_tts_manager is not None:
            kiwi_log("BARGE-IN", "Requested by listener", level="INFO")
        self._barge_in_requested = True
        if self._is_streaming:
            self._streaming_stop_event.set()
        self._cancel_idle_timer()
    
    def is_speaking(self) -> bool:
        """Возвращает True если Kiwi сейчас говорит."""
        return self._is_speaking or self._is_streaming or self._streaming_tts_manager is not None


def main():
    """Запуск сервиса с OpenClaw интеграцией."""
    
    # Setup crash protection
    if UTILS_AVAILABLE:
        setup_crash_protection()
    
    _setup_utf8_console_windows()
    
    log_func = kiwi_log if UTILS_AVAILABLE else print
    
    try:
        yaml_config = load_config_yaml("config.yaml")
        config = KiwiConfig.from_yaml(yaml_config)
        config.print_config_banner()
        
        service = KiwiServiceOpenClaw(config)
        
        service.start()
        log_func("KIWI", "\n" + "="*50)
        log_func("KIWI", "Сервис запущен!")
        log_func("KIWI", "="*50)
        log_func("KIWI", "Скажи 'Киви, привет!' чтобы начать")
        log_func("KIWI", "Ctrl+C для остановки\n")
        
        # Watchdog loop - check if daemon threads are alive and state timeouts
        while True:
            time.sleep(1)

            # Check state timeout — recover from stuck states
            try:
                if service._check_state_timeout():
                    current = service._get_state()
                    if current not in (DialogueState.IDLE, DialogueState.LISTENING):
                        log_func("WATCHDOG", f"State {current} timed out! Forcing recovery.", level="WARN")
                        # Попытка использовать накопленный текст вместо полной потери
                        finalized = False
                        try:
                            finalized = service._finalize_stalled_stream_from_accumulated(
                                time.time() - service._last_state_change,
                                f"state {current} timeout",
                            )
                        except Exception as e:
                            log_func("WATCHDOG", f"Finalize from accumulated failed: {e}", level="ERROR")

                        if not finalized:
                            # Cancel active OpenClaw request
                            try:
                                if hasattr(service, 'openclaw') and service.openclaw.is_processing():
                                    service.openclaw.cancel()
                            except Exception:
                                pass

                        # Stop streaming managers
                        if service._streaming_tts_manager:
                            try:
                                service._streaming_tts_manager.stop(graceful=False)
                            except Exception:
                                pass
                            service._streaming_tts_manager = None
                        if service._task_status_announcer:
                            try:
                                service._task_status_announcer.stop()
                            except Exception:
                                pass
                            service._task_status_announcer = None
                        service._is_speaking = False
                        service._is_streaming = False
                        service._set_state(DialogueState.IDLE)
                        service.listener.activate_dialog_mode()

                        if not finalized:
                            service.speak(
                                "Ответ занял слишком много времени. Повтори, пожалуйста.",
                                style="calm",
                            )
            except Exception as e:
                log_func("WATCHDOG", f"State timeout check error: {e}", level="ERROR")

            # Check if listener threads are alive and restart if dead
            if hasattr(service, 'listener') and service.listener:
                listener = service.listener
                if (listener._recording_thread and not listener._recording_thread.is_alive()):
                    log_func("WATCHDOG", "Recording thread died! Restarting...", level="ERROR")
                    try:
                        listener._recording_thread = threading.Thread(
                            target=listener._record_loop, daemon=True
                        )
                        listener._recording_thread.start()
                        log_func("WATCHDOG", "Recording thread restarted", level="INFO")
                    except Exception as e:
                        log_func("WATCHDOG", f"Failed to restart recording thread: {e}", level="ERROR")
                if (listener._processing_thread and not listener._processing_thread.is_alive()):
                    log_func("WATCHDOG", "Processing thread died! Restarting...", level="ERROR")
                    try:
                        listener._processing_thread = threading.Thread(
                            target=listener._process_loop, daemon=True
                        )
                        listener._processing_thread.start()
                        log_func("WATCHDOG", "Processing thread restarted", level="INFO")
                    except Exception as e:
                        log_func("WATCHDOG", f"Failed to restart processing thread: {e}", level="ERROR")
                    
    except KeyboardInterrupt:
        log_func("KIWI", "\n[BYE] Останавливаюсь...")
        service.stop()
    except Exception as e:
        log_func("CRITICAL", f"Unhandled exception in main: {e}", level="ERROR")
        log_func("CRITICAL", traceback.format_exc(), level="ERROR")
        try:
            service.stop()
        except:
            pass
        raise


if __name__ == "__main__":
    main()
