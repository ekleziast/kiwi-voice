#!/usr/bin/env python3
"""
Kiwi Voice Service - OpenClaw Integration.

Main orchestrator: KiwiServiceOpenClaw class + main() entry point.
Mixin modules: audio_playback, tts_speech, stream_watchdog,
llm_callbacks, dialogue_pipeline.
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
import traceback
from typing import Optional
import queue

import sounddevice as sd

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

# Mixins
from kiwi.mixins import (
    AudioPlaybackMixin,
    TTSSpeechMixin,
    StreamWatchdogMixin,
    LLMCallbacksMixin,
    DialoguePipelineMixin,
)
from kiwi.mixins.dialogue_pipeline import CommandContext  # re-export


class KiwiServiceOpenClaw(
    AudioPlaybackMixin,
    TTSSpeechMixin,
    StreamWatchdogMixin,
    LLMCallbacksMixin,
    DialoguePipelineMixin,
):
    """Главный сервис голосового ассистента Киви с OpenClaw интеграцией."""

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
        self._pending_owner_approval: Optional[dict] = None
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
        if self.config.ws_enabled:
            try:
                ws_client = OpenClawWebSocket(
                    config=self.config,
                    on_token=self._on_llm_token,
                    on_complete=self._on_llm_complete,
                    on_activity=self._on_agent_activity,
                    log_func=kiwi_log if UTILS_AVAILABLE else print,
                )
                if ws_client.connect():
                    kiwi_log("KIWI", f"WebSocket connected to {self.config.ws_host}:{self.config.ws_port}", "INFO")
                    self._use_websocket = True
                    return ws_client
                else:
                    kiwi_log("KIWI", "WebSocket connection failed, will fallback to CLI", "WARN")
                    ws_client.close()
            except Exception as e:
                kiwi_log("KIWI", f"WebSocket initialization error: {e}", "ERROR")

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

        if EVENT_BUS_AVAILABLE:
            from kiwi.event_bus import EventType
            get_event_bus().publish(
                EventType.STATE_CHANGED,
                {'old_state': old_state, 'new_state': new_state, 'timeout': timeout},
                source='kiwi_service'
            )

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

    # === IDLE / BARGE-IN ===

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

    # === LIFECYCLE ===

    def start(self):
        """Запускает сервис с приветственным звуком и сообщением."""
        if self.is_running:
            return

        with open(os.path.join(LOGS_DIR, 'kiwi_startup.log'), 'a', encoding='utf-8') as f:
            f.write('[START] KiwiServiceOpenClaw.start() called\n')

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
        if EVENT_BUS_AVAILABLE:
            from kiwi.event_bus import EventType
            get_event_bus().publish(
                EventType.SYSTEM_SHUTDOWN,
                {},
                source='kiwi_service'
            )
            get_event_bus().stop()

        self.is_running = False

        self._barge_in_requested = True
        self._streaming_stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass

        if self._streaming_tts_manager:
            try:
                self._streaming_tts_manager.stop(graceful=False)
            except Exception:
                pass
            self._streaming_tts_manager = None

        if self._task_status_announcer:
            try:
                self._task_status_announcer.stop()
            except Exception:
                pass
            self._task_status_announcer = None

        self._cancel_idle_timer()

        self.listener.stop()
        kiwi_log("KIWI", "Kiwi Voice Service stopped", level="INFO")


def main():
    """Запуск сервиса с OpenClaw интеграцией."""

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
                        finalized = False
                        try:
                            finalized = service._finalize_stalled_stream_from_accumulated(
                                time.time() - service._last_state_change,
                                f"state {current} timeout",
                            )
                        except Exception as e:
                            log_func("WATCHDOG", f"Finalize from accumulated failed: {e}", level="ERROR")

                        if not finalized:
                            try:
                                if hasattr(service, 'openclaw') and service.openclaw.is_processing():
                                    service.openclaw.cancel()
                            except Exception:
                                pass

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
