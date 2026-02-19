#!/usr/bin/env python3
"""OpenClaw WebSocket client for Kiwi Voice (Gateway v3 protocol)."""

import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

try:
    from kiwi.utils import kiwi_log
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    def kiwi_log(tag: str, message: str, level: str = "INFO"):
        print(f"[{tag}] {message}", flush=True)

from kiwi.config_loader import KiwiConfig


class OpenClawWebSocket:
    """WebSocket клиент для streaming общения с OpenClaw Gateway v3.

    Протокол:
    1. Сервер шлёт connect.challenge event
    2. Клиент отвечает connect request с ConnectParams
    3. Сервер отвечает hello-ok
    4. Клиент отправляет chat.send requests
    5. Сервер шлёт chat events (delta/final/error/aborted)

    Поддерживает:
    - Протокол OpenClaw Gateway v3
    - Автоматическое переподключение с настраиваемым интервалом
    - Логирование через kiwi_log
    - Fallback на CLI при недоступности WebSocket
    """

    PROTOCOL_VERSION = 3

    # Допустимые client.id (enum)
    VALID_CLIENT_IDS = {
        "webchat-ui", "openclaw-control-ui", "webchat", "cli", "gateway-client",
        "openclaw-macos", "openclaw-ios", "openclaw-android", "node-host", "test",
        "fingerprint", "openclaw-probe"
    }

    # Допустимые client.mode (enum)
    VALID_CLIENT_MODES = {
        "webchat", "cli", "ui", "backend", "node", "probe", "test"
    }

    def __init__(
        self,
        config: KiwiConfig,
        on_token: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[str], None]] = None,
        on_activity: Optional[Callable[[dict], None]] = None,
        log_func: Optional[Callable] = None,
    ):
        self.config = config
        self.on_token = on_token
        self.on_complete = on_complete
        self.on_activity = on_activity
        self._log = log_func if log_func else (kiwi_log if UTILS_AVAILABLE else print)

        # WebSocket state
        self._ws = None
        self._ws_thread: Optional[threading.Thread] = None
        self._is_connected = False       # TCP connected
        self._is_authenticated = False   # Handshake complete (hello-ok received)
        self._is_streaming = False
        self._is_processing = False      # Для совместимости с OpenClawCLI
        self._accumulated_text = ""
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_ws_recv_ts = 0.0  # timestamp of last received WS message

        # Reconnection state
        self._reconnect_attempts = 0
        self._last_connect_time = 0.0
        self._reconnect_thread: Optional[threading.Thread] = None

        # Protocol v3: pending requests (id → threading.Event + result)
        self._pending_requests: Dict[str, dict] = {}
        self._pending_lock = threading.Lock()

        # Gateway token
        self._gateway_token = self._load_gateway_token()

        # Device identity (Ed25519 key pair for gateway device auth)
        self._device_identity = self._load_or_create_device_identity()

        # Session key format: "agent:{agent_id}:{session_id}"
        self._session_key = f"agent:{config.openclaw_session_id}:{config.openclaw_session_id}"

        # Current chat run tracking
        self._current_run_id: Optional[str] = None
        self._seen_final_run_ids: set[str] = set()

        # Deferred final: debounce lifecycle:end to support multi-wave agent responses.
        # Agent may complete multiple runs (tool calls, research steps) within a single
        # user request — each run sends lifecycle:end but only the LAST one is truly final.
        self._deferred_final_timer: Optional[threading.Timer] = None
        self._deferred_final_info: Optional[dict] = None
        self._deferred_final_lock = threading.Lock()
        self._lifecycle_final_debounce_s = 2.5

        # Задержка перед отправкой накопленного текста в TTS (debounce)
        self._tts_debounce_ms = 150
        self._last_send_time = 0.0
        self._pending_text = ""

        # Ответ, полученный через WebSocket
        self._full_response = ""
        self._response_event = threading.Event()
        # Режим callback'ов: "streaming" или "blocking" (для OpenClawCLI-совместимого chat()).
        self._callback_mode = "streaming"

        # Auth event — ждём завершения handshake
        self._auth_event = threading.Event()

        self._log("OPENCLAW-WS", f"Initialized with host={config.ws_host}, port={config.ws_port}, session_key={self._session_key}")

    def _log_ws(self, message: str, level: str = "INFO"):
        """Внутренний метод логирования."""
        if UTILS_AVAILABLE:
            kiwi_log("OPENCLAW-WS", message, level)
        else:
            print(f"[OPENCLAW-WS] {message}", flush=True)

    def _touch_stream_progress(self, stream: str):
        """Сообщает внешнему watchdog, что по run пришла живая активность.

        Важно: используем пустое сообщение, чтобы не озвучивать reasoning/tool детали.
        """
        if not self.on_activity:
            return
        try:
            self.on_activity({
                "type": "stream-progress",
                "stream": stream,
                "message": "",
            })
        except Exception as e:
            self._log_ws(f"Progress callback error ({stream}): {e}", "DEBUG")

    def _load_gateway_token(self) -> str:
        """Загружает gateway token из ~/.openclaw/openclaw.json или env."""
        # 1. Из переменной окружения
        token = os.getenv("OPENCLAW_GATEWAY_TOKEN")
        if token:
            self._log_ws(f"Gateway token loaded from env", "DEBUG")
            return token

        # 2. Из ~/.openclaw/openclaw.json
        config_path = os.path.join(os.path.expanduser("~"), ".openclaw", "openclaw.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    oc_config = json.load(f)
                token = oc_config.get("gateway", {}).get("auth", {}).get("token", "")
                if token:
                    self._log_ws(f"Gateway token loaded from {config_path}", "DEBUG")
                    return token
        except Exception as e:
            self._log_ws(f"Failed to read gateway token from {config_path}: {e}", "WARN")

        self._log_ws("No gateway token found! Auth will fail.", "ERROR")
        return ""

    # --- Device Identity (Ed25519) ---

    def _load_or_create_device_identity(self) -> dict:
        """Загружает или генерирует Ed25519 ключевую пару для device auth.

        Ключи сохраняются в ~/.openclaw/workspace/skills/kiwi-voice/device-identity.json
        чтобы device ID оставался постоянным между перезапусками.
        """
        identity_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "device-identity.json",
        )

        # Попробуем загрузить существующую identity
        if os.path.exists(identity_path):
            try:
                with open(identity_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("deviceId") and data.get("publicKey") and data.get("privateKey"):
                    self._log_ws(f"Device identity loaded (id={data['deviceId'][:12]}...)", "DEBUG")
                    return data
            except Exception as e:
                self._log_ws(f"Failed to load device identity: {e}", "WARN")

        # Генерируем новую
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        priv_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

        pub_b64 = base64.urlsafe_b64encode(pub_bytes).rstrip(b"=").decode("ascii")
        priv_b64 = base64.urlsafe_b64encode(priv_bytes).rstrip(b"=").decode("ascii")
        device_id = hashlib.sha256(pub_bytes).hexdigest()

        data = {"deviceId": device_id, "publicKey": pub_b64, "privateKey": priv_b64}

        try:
            os.makedirs(os.path.dirname(identity_path), exist_ok=True)
            with open(identity_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._log_ws(f"Device identity created (id={device_id[:12]}...)", "INFO")
        except Exception as e:
            self._log_ws(f"Failed to save device identity: {e}", "WARN")

        return data

    def _build_device_auth(self, nonce: str) -> dict:
        """Строит device auth объект для connect request.

        Формат payload (v2):
          v2|deviceId|clientId|clientMode|role|scopes|signedAtMs|token|nonce
        """
        identity = self._device_identity
        signed_at_ms = int(time.time() * 1000)

        scopes_str = ",".join(["operator.admin"])
        payload_parts = [
            "v2",
            identity["deviceId"],
            "gateway-client",   # client.id
            "backend",          # client.mode
            "operator",         # role
            scopes_str,
            str(signed_at_ms),
            self._gateway_token,
            nonce,
        ]
        payload = "|".join(payload_parts)

        # Подписываем Ed25519
        priv_bytes = base64.urlsafe_b64decode(identity["privateKey"] + "==")
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(priv_bytes)
        signature = private_key.sign(payload.encode("utf-8"))
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")

        return {
            "id": identity["deviceId"],
            "publicKey": identity["publicKey"],
            "signature": sig_b64,
            "signedAt": signed_at_ms,
            "nonce": nonce,
        }

    def _get_ws_url(self) -> str:
        """Формирует URL для WebSocket подключения (без пути!)."""
        return f"ws://{self.config.ws_host}:{self.config.ws_port}"

    def connect(self) -> bool:
        """Устанавливает WebSocket соединение с OpenClaw Gateway v3.

        Протокол:
        1. TCP connect → on_open
        2. Ждём connect.challenge event от сервера
        3. Отправляем connect request с ConnectParams
        4. Ждём hello-ok response

        Returns:
            True если подключение и аутентификация успешны, False иначе
        """
        if self._is_authenticated:
            return True

        if self._stop_event.is_set():
            self._log_ws("Cannot connect: stop event is set", "WARN")
            return False

        # Сбрасываем состояние аутентификации
        self._is_authenticated = False
        self._auth_event.clear()

        try:
            import websocket

            ws_url = self._get_ws_url()
            self._log_ws(f"Connecting to {ws_url} (protocol v{self.PROTOCOL_VERSION})...")

            def on_message(ws, message):
                try:
                    self._handle_message(message)
                except Exception as e:
                    self._log_ws(f"Error handling message: {e}", "ERROR")
                    import traceback
                    self._log_ws(traceback.format_exc(), "ERROR")

            def on_error(ws, error):
                self._log_ws(f"WebSocket error: {error}", "ERROR")
                self._is_connected = False
                self._is_authenticated = False
                if not self._stop_event.is_set():
                    self._fail_active_request(f"websocket error: {error}")

            def on_close(ws, close_status_code, close_msg):
                self._log_ws(f"Connection closed: {close_status_code} - {close_msg}", "WARN")
                self._is_connected = False
                self._is_authenticated = False
                if not self._stop_event.is_set():
                    reason = f"connection closed: code={close_status_code}, msg={close_msg}"
                    self._fail_active_request(reason)
                # Запускаем переподключение
                if close_status_code not in (1000,):
                    self._schedule_reconnect()

            def on_open(ws):
                self._log_ws("TCP connected, waiting for connect.challenge...", "INFO")
                self._is_connected = True
                self._reconnect_attempts = 0
                self._last_connect_time = time.time()
                # НЕ отправляем handshake сразу! Ждём connect.challenge от сервера.

            # WebSocket без кастомных заголовков (протокол v3 не требует)
            self._ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            # Запускаем WebSocket в отдельном потоке
            self._ws_thread = threading.Thread(
                target=lambda: self._ws.run_forever(
                    ping_interval=max(0.0, float(self.config.ws_ping_interval)),
                    ping_timeout=max(1.0, float(self.config.ws_ping_timeout)),
                ),
                daemon=True
            )
            self._ws_thread.start()

            # Ждём полной аутентификации (TCP + challenge + hello-ok)
            auth_timeout = 15.0
            self._log_ws(f"Waiting for authentication (timeout={auth_timeout}s)...", "DEBUG")

            if self._auth_event.wait(timeout=auth_timeout):
                self._log_ws("Fully authenticated with Gateway v3", "INFO")
                return True
            else:
                # Проверяем, подключились ли хотя бы по TCP
                if self._is_connected:
                    self._log_ws(f"Auth timeout after {auth_timeout}s (TCP ok, but no hello-ok)", "WARN")
                else:
                    self._log_ws(f"Connection timeout after {auth_timeout}s", "WARN")
                return False

        except ImportError:
            self._log_ws("websocket-client not installed. Install with: pip install websocket-client", "ERROR")
            return False
        except Exception as e:
            self._log_ws(f"Connection failed: {e}", "ERROR")
            return False

    def _schedule_reconnect(self):
        """Планирует переподключение в отдельном потоке."""
        if self._stop_event.is_set():
            return

        if self._is_authenticated:
            return  # Already connected and authenticated, no reconnect needed

        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return  # Уже переподключаемся

        self._reconnect_attempts += 1
        if self._reconnect_attempts > self.config.ws_max_reconnect_attempts:
            self._log_ws(f"Max reconnect attempts ({self.config.ws_max_reconnect_attempts}) reached. Giving up.", "ERROR")
            return

        def reconnect_worker():
            delay = self.config.ws_reconnect_interval
            self._log_ws(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts}/{self.config.ws_max_reconnect_attempts})...", "INFO")
            time.sleep(delay)
            if not self._stop_event.is_set() and not self._is_authenticated:
                self.connect()

        self._reconnect_thread = threading.Thread(target=reconnect_worker, daemon=True)
        self._reconnect_thread.start()

    def is_ws_alive(self, threshold_s: float = 15.0) -> bool:
        """True if a WS message was received within *threshold_s* seconds."""
        if not self._is_authenticated:
            return False
        if self._last_ws_recv_ts <= 0:
            return False
        return (time.time() - self._last_ws_recv_ts) < threshold_s

    def force_reconnect(self, reason: str = "forced") -> bool:
        """Close current WS and reconnect synchronously (best-effort, timeout 10s)."""
        self._log_ws(f"Force reconnect: {reason}", "WARNING")
        self._is_authenticated = False
        self._is_connected = False

        # Close existing socket
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

        # Wait for WS thread to die
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=3.0)

        # Reconnect (blocking, up to ~15s with auth wait)
        ok = self.connect()
        if ok:
            self._log_ws("Force reconnect succeeded", "INFO")
        else:
            self._log_ws("Force reconnect failed", "ERROR")
        return ok

    def _handle_message(self, message: str):
        """Обрабатывает сообщения от WebSocket по протоколу Gateway v3.

        Типы фреймов:
        - "event" → connect.challenge, chat events, etc.
        - "res"   → ответы на наши requests (connect, chat.send, chat.abort)
        """
        try:
            self._last_ws_recv_ts = time.time()
            data = json.loads(message)
            msg_type = data.get("type", "")

            if msg_type == "event":
                self._handle_event(data)

            elif msg_type == "res":
                self._handle_response(data)

            else:
                self._log_ws(f"Unknown frame type: {msg_type} | {message[:100]}", "WARN")

        except json.JSONDecodeError:
            self._log_ws(f"Non-JSON message received: {message[:80]}...", "WARN")
        except Exception as e:
            self._log_ws(f"Message handling error: {e}", "ERROR")

    def _fail_active_request(self, reason: str):
        """Завершает активный запрос ошибкой, чтобы не зависать при обрыве WS."""
        if not (self._is_processing or self._is_streaming):
            return

        self._log_ws(f"Fail active request: {reason}", "WARN")
        self._is_streaming = False
        self._is_processing = False
        self._current_run_id = None

        acquired = self._buffer_lock.acquire(timeout=3.0)
        if acquired:
            try:
                if not self._full_response:
                    self._full_response = (
                        "Извини, соединение с OpenClaw прервалось. "
                        "Повтори запрос."
                    )
            finally:
                self._buffer_lock.release()
        else:
            self._log_ws("_fail_active_request: _buffer_lock timeout (3s)", "WARNING")
            if not self._full_response:
                self._full_response = (
                    "Извини, соединение с OpenClaw прервалось. "
                    "Повтори запрос."
                )

        self._response_event.set()

        # В streaming-режиме подаём completion, чтобы остановить StreamingTTSManager.
        if self._callback_mode == "streaming" and self.on_complete:
            try:
                self.on_complete(self._full_response)
            except Exception as e:
                self._log_ws(f"on_complete after failure error: {e}", "DEBUG")

    def _handle_event(self, data: dict):
        """Обрабатывает event-фреймы от сервера."""
        event_name = str(data.get("event", "")).strip().lower()
        payload = data.get("payload", {})
        seq = data.get("seq", -1)

        if event_name == "connect.challenge":
            # Шаг 1 протокола: сервер прислал challenge
            nonce = payload.get("nonce", "")
            ts = payload.get("ts", 0)
            self._log_ws(f"Received connect.challenge (nonce={nonce[:16]}..., ts={ts})", "INFO")
            self._send_connect(nonce)

        elif event_name == "chat":
            # Legacy/normalized chat event with payload.state
            self._handle_chat_event(payload)
        elif event_name == "agent":
            # Native agent stream event: payload.stream + payload.data
            self._handle_agent_event(payload)

        else:
            self._log_ws(f"Event: {event_name} (seq={seq})", "DEBUG")

    def _handle_response(self, data: dict):
        """Обрабатывает response-фреймы (ответы на наши requests)."""
        req_id = data.get("id", "")
        ok = data.get("ok", False)
        payload = data.get("payload", {})

        # Проверяем, есть ли pending request с таким id
        with self._pending_lock:
            pending = self._pending_requests.get(req_id)

        if pending:
            # Сохраняем результат и сигнализируем
            pending["ok"] = ok
            pending["payload"] = payload
            pending["event"].set()

            if ok:
                payload_type = payload.get("type", "")
                if payload_type == "hello-ok":
                    # Handshake завершён успешно!
                    protocol = payload.get("protocol", 0)
                    self._log_ws(f"Authenticated! Protocol v{protocol}, hello-ok received", "INFO")
                    self._is_authenticated = True
                    self._reconnect_attempts = 0
                    self._auth_event.set()
                else:
                    self._log_ws(f"Response OK for {req_id}: {payload_type or str(payload)[:60]}", "DEBUG")
                    if pending.get("method") == "chat.send":
                        run_id = payload.get("runId")
                        if run_id:
                            # Only set if not already set by an arriving event.
                            # Events can arrive BEFORE the chat.send response,
                            # and their runId is the source of truth.
                            if not self._current_run_id:
                                self._current_run_id = run_id
                                self._log_ws(f"Active runId set from chat.send response: {run_id}", "DEBUG")
                            elif self._current_run_id != run_id:
                                self._log_ws(
                                    f"runId mismatch: response={run_id}, active={self._current_run_id}. "
                                    f"Keeping active (set by event).",
                                    "WARN",
                                )
            else:
                error = payload.get("error", payload.get("message", str(payload)))
                self._log_ws(f"Response ERROR for {req_id}: {error}", "ERROR")
        else:
            self._log_ws(f"Response for unknown request {req_id}: ok={ok}", "WARN")

    def _extract_text_from_delta(self, content: str) -> str:
        """Извлекает текст из delta content, который может быть строковым dict'ом или списком.

        ЧИСТЫЙ REGEX — без ast.literal_eval для избежания проблем с форматированием.

        Обрабатывает случаи:
        - [{'type': 'text', 'text': '...'}] (список с одним dict)
        - {'type': 'text', 'text': '...'} (одиночный dict)
        - Конкатенацию нескольких dict'ов: {'type': 'text', 'text': 'С'}{'type': 'text', 'text': 'О'}
        - Смешанный контент: текст + dict'ы
        """
        if not isinstance(content, str):
            return str(content) if content else ""

        stripped = content.strip()
        if not stripped:
            return ""

        # Если это обычный текст без паттернов dict — возвращаем как есть
        if not (("'text'" in stripped or '"text"' in stripped) and
                (stripped.startswith('{') or stripped.startswith('['))):
            return content

        import re

        # Случай 1: Список с dict'ами [{'type': 'text', 'text': '...'}]
        # Используем regex для извлечения всех dict'ов из списка
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

        # Случай 2 & 3: Одиночный dict или конкатенация dict'ов
        # Ищем все вхождения 'text': '...' (с одинарными кавычками)
        matches = re.findall(r"'text':\s*'([^']*?)'", content)
        if matches:
            result = "".join(matches)
            if result:
                return result

        # Ищем с двойными кавычками "text": "..."
        matches = re.findall(r'"text":\s*"([^"]*?)"', content)
        if matches:
            result = "".join(matches)
            if result:
                return result

        # Случай 4: Разбиваем по }{ и ищем text в каждой части
        if '}{' in content:
            parts = content.split('}{')
            texts = []
            for i, part in enumerate(parts):
                # Добавляем скобки обратно
                if i == 0:
                    part = part + '}'
                elif i == len(parts) - 1:
                    part = '{' + part
                else:
                    part = '{' + part + '}'

                # Ищем text с помощью regex (без ast.literal_eval)
                text_match = re.search(r"'text':\s*'([^']*?)'", part)
                if text_match:
                    texts.append(text_match.group(1))
                else:
                    text_match = re.search(r'"text":\s*"([^"]*?)"', part)
                    if text_match:
                        texts.append(text_match.group(1))

            if texts:
                return "".join(texts)

        # Если ничего не сработало — возвращаем как есть
        return content

    def _normalize_chat_content(self, content) -> str:
        """Нормализует chat content (dict/list/str) в plain text."""
        if content is None:
            return ""

        # content может быть dict {'type': 'text', 'text': '...'}
        if isinstance(content, dict):
            content = content.get('text', content.get('content', ""))
        # content может быть list (например, [{'type': 'text', 'text': '...'}])
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    texts.append(item.get('text', item.get('content', "")))
                else:
                    texts.append(str(item))
            content = "".join(t for t in texts if t is not None)

        if isinstance(content, str):
            return self._extract_text_from_delta(content)
        return str(content) if content else ""

    def _extract_text_from_payload(self, payload: dict, state: str) -> str:
        """Извлекает текст из разных форматов payload (устойчиво к изменениям протокола)."""
        candidates = []

        # Основной путь (текущая схема)
        message_data = payload.get("message", {})
        if isinstance(message_data, dict) and "content" in message_data:
            candidates.append(message_data.get("content"))

        # Fallback-поля, которые встречаются в разных реализациях chat событий
        for key in ("content", "text", "delta", "output", "answer", "response", "final", "result", "data"):
            if key in payload:
                candidates.append(payload.get(key))

        normalized = []
        for item in candidates:
            text = self._normalize_chat_content(item).strip()
            if text:
                normalized.append(text)

        if not normalized:
            return ""

        # Для delta/final выбираем самый содержательный вариант
        if state in ("delta", "final"):
            return max(normalized, key=len)

        return normalized[0]

    def _emit_activity(self, activity_type: str, message: str, details: Optional[dict] = None):
        """Передаёт событие активности в сервис, если подключен callback."""
        if not self.on_activity or not message:
            return
        try:
            self.on_activity({
                "type": activity_type,
                "message": message,
                "details": details or {},
            })
        except Exception as e:
            self._log_ws(f"Activity callback error: {e}", "DEBUG")

    def _extract_tool_name(self, data: dict) -> str:
        tool = data.get("tool") or data.get("name") or data.get("toolName") or data.get("id")
        if isinstance(tool, dict):
            tool = tool.get("name") or tool.get("id")
        if not tool:
            tool_info = data.get("toolCall") or data.get("call") or {}
            if isinstance(tool_info, dict):
                tool = tool_info.get("name") or tool_info.get("tool")
        return str(tool or "").strip()

    def _extract_tool_command(self, data: dict) -> str:
        candidates = [data.get("command"), data.get("cmd"), data.get("script")]

        for key in ("input", "args", "parameters", "params"):
            value = data.get(key)
            if isinstance(value, dict):
                candidates.append(value.get("command"))
                candidates.append(value.get("cmd"))
                candidates.append(value.get("script"))

        tool_call = data.get("toolCall") or data.get("call")
        if isinstance(tool_call, dict):
            args = tool_call.get("args") or tool_call.get("arguments")
            if isinstance(args, dict):
                candidates.append(args.get("command"))
                candidates.append(args.get("cmd"))
                candidates.append(args.get("script"))
            elif isinstance(args, str):
                candidates.append(args)

        for item in candidates:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return ""

    def _describe_tool_activity(self, data: dict) -> str:
        command = self._extract_tool_command(data)
        tool_name = self._extract_tool_name(data).lower()
        phase = str(data.get("phase", data.get("status", data.get("state", "")))).lower()

        if phase in ("error", "failed", "fail"):
            return "Инструмент вернул ошибку, проверяю причину."
        if phase in ("end", "done", "complete", "completed", "success", "ok"):
            return ""

        if command:
            cmd = command.lower()
            if re.search(r"\bcd\b", cmd) or "set-location" in cmd:
                return "Открываю нужную папку проекта."
            if "get-childitem" in cmd or re.search(r"\brg\b", cmd) or " --files" in cmd or re.search(r"\bls\b", cmd):
                return "Смотрю структуру проекта и ищу нужные файлы."
            if "get-content" in cmd or re.search(r"\bcat\b", cmd) or re.search(r"\bsed\b", cmd):
                match = re.search(r"(?:get-content|cat)\s+([^\s|;]+)", command, re.IGNORECASE)
                if match:
                    path = match.group(1).strip("'\"")
                    return f"Анализирую файл {path}."
                return "Читаю и анализирую код файлов."
            if "select-string" in cmd or re.search(r"\brg\s+-n\b", cmd):
                return "Ищу нужные места в коде."
            if "pytest" in cmd or "py_compile" in cmd or "npm test" in cmd or "cargo test" in cmd:
                return "Проверяю код и прогоняю проверки."
            if "apply_patch" in cmd or "*** begin patch" in cmd:
                return "Вношу изменения в код."
            if re.search(r"\bgit\s+", cmd):
                return "Проверяю изменения в репозитории."
            return "Выполняю шаги в проекте."

        if "shell_command" in tool_name:
            return "Выполняю команду в терминале."
        if "apply_patch" in tool_name:
            return "Вношу изменения в код."
        if "read" in tool_name or "open" in tool_name:
            return "Читаю файлы проекта."
        if "search" in tool_name or "find" in tool_name:
            return "Ищу нужную информацию в коде."

        return "Продолжаю работу над задачей."

    def _handle_agent_event(self, payload: dict):
        """Преобразует native agent stream events в chat-like события для единого пайплайна."""
        run_id = payload.get("runId", "")
        session_key = payload.get("sessionKey", "")
        stream = str(payload.get("stream", "")).strip().lower()
        data = payload.get("data", {})
        seq = payload.get("seq", -1)

        # Игнорируем события от чужих сессий.
        if session_key and session_key != self._session_key:
            self._log_ws(
                f"Ignoring agent event for foreign sessionKey={session_key} "
                f"(mine={self._session_key}, stream={stream}, runId={run_id[:12] if run_id else 'none'})",
                "WARN" if stream == "lifecycle" else "DEBUG",
            )
            return

        if not isinstance(data, dict):
            data = {"value": data}

        if not stream:
            self._log_ws(
                f"Agent event without stream. Keys: {list(payload.keys())[:10]}",
                "DEBUG"
            )
            return

        # Некоторые рантаймы стримят рассуждения отдельным потоком (thinking/reasoning),
        # и между текстовыми assistant-delta может проходить заметное время.
        # Отмечаем такую активность, чтобы watchdog не делал ложный stall/retry.
        if stream in ("thinking", "reasoning", "compaction"):
            self._cancel_deferred_final()  # agent is still working
            self._touch_stream_progress(stream)

        if stream == "assistant":
            # Agent is producing text — cancel any pending deferred final
            # from a previous lifecycle:end (multi-wave response continues).
            self._cancel_deferred_final()

            # В OpenClaw canonical поле для chat-bridge — data.text (см. server-chat.ts).
            # data.delta используем только как fallback.
            assistant_content = data.get("text", data.get("content", ""))
            if assistant_content is None or assistant_content == "":
                assistant_content = data.get("delta", "")

            synthetic = {
                "runId": run_id,
                "sessionKey": session_key,
                "seq": seq,
                "state": "delta",
                "message": {"content": assistant_content},
            }
            self._handle_chat_event(synthetic)
            return

        if stream == "lifecycle":
            phase = str(data.get("phase", "")).strip().lower()
            # Любое lifecycle-событие — признак активности, обновляем watchdog.
            self._touch_stream_progress(f"lifecycle:{phase}")
            # Suppress immediate start announcement for short requests.
            if phase in ("thinking", "plan", "planning"):
                # Agent starting a new thinking step — cancel pending final
                self._cancel_deferred_final()
                self._emit_activity("lifecycle", "Планирую шаги решения.", {"phase": phase})

            if phase in ("end", "done", "complete", "completed", "finish", "finished"):
                # Don't fire final immediately — the agent may continue with
                # more runs (tool calls, research steps).  Use deferred final
                # with debounce so that multi-wave responses are kept intact.
                final_content = data.get("text", data.get("content", ""))
                self._current_run_id = None  # allow next wave's events through
                self._schedule_deferred_final(run_id, session_key, seq, final_content)
                return

            if phase in ("error", "failed", "fail"):
                self._cancel_deferred_final()  # error overrides any pending final
                self._emit_activity("lifecycle", "Возникла ошибка, проверяю что случилось.", {"phase": phase})
                err = data.get("error") or data.get("message") or data.get("reason") or "Unknown lifecycle error"
                self._handle_chat_event({
                    "runId": run_id,
                    "sessionKey": session_key,
                    "seq": seq,
                    "state": "error",
                    "errorMessage": str(err),
                })
                return

            if phase in ("aborted", "cancelled", "canceled", "stop", "stopped"):
                self._cancel_deferred_final()  # abort overrides any pending final
                self._handle_chat_event({
                    "runId": run_id,
                    "sessionKey": session_key,
                    "seq": seq,
                    "state": "aborted",
                })
                return

            self._log_ws(f"Agent lifecycle phase: {phase or '<empty>'}", "DEBUG")
            return

        if stream == "error":
            err = data.get("error") or data.get("message") or data.get("reason") or "Unknown agent stream error"
            self._handle_chat_event({
                "runId": run_id,
                "sessionKey": session_key,
                "seq": seq,
                "state": "error",
                "errorMessage": str(err),
            })
            return

        if stream == "tool":
            # Agent is calling a tool — cancel any pending deferred final
            # (multi-wave response continues with tool invocation).
            self._cancel_deferred_final()

            # Обновляем watchdog — tool вызовы означают, что LLM работает,
            # даже если текстовых токенов ещё нет.
            self._touch_stream_progress(stream)
            tool_msg = self._describe_tool_activity(data)
            if tool_msg:
                self._emit_activity("tool", tool_msg, {"stream": stream})
            return

        # tool/compaction/reasoning и прочие служебные потоки в голосовой ответ не конвертируем.
        if stream not in ("compaction",):
            self._log_ws(f"Agent stream ignored: {stream}", "DEBUG")

    # ------------------------------------------------------------------
    # Deferred final: debounce lifecycle:end for multi-wave agent responses
    # ------------------------------------------------------------------

    def _schedule_deferred_final(self, run_id, session_key, seq, content):
        """Schedule a final event after debounce to support multi-wave responses.

        If the agent continues producing text (new deltas, tool calls, etc.)
        within the debounce window, the timer is cancelled.  Only when the
        agent is truly silent the final is fired.
        """
        with self._deferred_final_lock:
            # Cancel any existing timer
            if self._deferred_final_timer is not None:
                self._deferred_final_timer.cancel()
                self._deferred_final_timer = None

            self._deferred_final_info = {
                "run_id": run_id,
                "session_key": session_key,
                "seq": seq,
                "content": content,
            }

            self._deferred_final_timer = threading.Timer(
                self._lifecycle_final_debounce_s,
                self._fire_deferred_final,
            )
            self._deferred_final_timer.daemon = True
            self._deferred_final_timer.start()

        self._log_ws(
            f"Deferred final scheduled ({self._lifecycle_final_debounce_s}s debounce, "
            f"runId={run_id[:12] if run_id else 'none'})",
            "DEBUG",
        )

    def _cancel_deferred_final(self):
        """Cancel pending deferred final — agent is still active."""
        with self._deferred_final_lock:
            if self._deferred_final_timer is None:
                return
            self._deferred_final_timer.cancel()
            self._deferred_final_timer = None
            self._deferred_final_info = None
        self._log_ws("Deferred final cancelled (agent still active)", "DEBUG")

    def _fire_deferred_final(self):
        """Called by the debounce timer — agent has been silent, emit final."""
        with self._deferred_final_lock:
            info = self._deferred_final_info
            self._deferred_final_info = None
            self._deferred_final_timer = None

        if info is None:
            return

        self._log_ws(
            f"Deferred final fired (runId={info['run_id'][:12] if info.get('run_id') else 'none'})",
            "INFO",
        )

        # Use accumulated text as the final content (has ALL waves)
        with self._buffer_lock:
            full_text = self._accumulated_text or self._full_response or ""

        synthetic = {
            "runId": info["run_id"],
            "sessionKey": info["session_key"],
            "seq": info["seq"],
            "state": "final",
        }
        if full_text:
            synthetic["message"] = {"content": full_text}

        self._handle_chat_event(synthetic)

    def _handle_chat_event(self, payload: dict):
        """Обрабатывает chat events (delta/final/error/aborted)."""
        # Некоторые шлюзы используют state, некоторые status
        raw_state = payload.get("state", payload.get("status", ""))
        state = str(raw_state).strip().lower() if raw_state is not None else ""
        state_alias = {
            "completed": "final",
            "complete": "final",
            "done": "final",
            "finish": "final",
            "finished": "final",
            "chunk": "delta",
            "partial": "delta",
            "fail": "error",
            "cancelled": "aborted",
            "canceled": "aborted",
        }
        state = state_alias.get(state, state)

        # Игнорируем пустые state (промежуточные события от Gateway)
        if not state:
            if payload:
                self._log_ws(
                    f"Chat event without state/status. Keys: {list(payload.keys())[:10]}",
                    "DEBUG"
                )
            return

        run_id = payload.get("runId", "")
        session_key = payload.get("sessionKey", "")
        content = self._extract_text_from_payload(payload, state)

        # Игнорируем события от чужих сессий (другие агенты на том же Gateway).
        if session_key and session_key != self._session_key:
            self._log_ws(
                f"Ignoring event for foreign sessionKey={session_key} (mine={self._session_key}, state={state})",
                "DEBUG",
            )
            return

        # Игнорируем события от предыдущих/чужих runId, если уже знаем активный runId.
        if run_id and self._current_run_id and run_id != self._current_run_id:
            self._log_ws(
                f"Ignoring event for stale runId={run_id} (active={self._current_run_id}, state={state})",
                "DEBUG",
            )
            return

        # === ДИАГНОСТИКА: логируем raw payload для terminal событий ===
        if state == "final":
            self._log_ws(f"RAW final payload: {json.dumps(payload, ensure_ascii=False)[:500]}...", "DEBUG")

        # === ДИАГНОСТИКА: логируем raw content для отладки ===
        if state == "delta" and content:
            raw_preview = str(content)[:100].replace('\n', ' ')
            self._log_ws(f"Raw delta content: {raw_preview}...", "DEBUG")

        # === ДИАГНОСТИКА: логируем результат очистки ===
        if state == "delta" and content:
            cleaned_preview = content[:100].replace('\n', ' ')
            self._log_ws(f"Cleaned delta content: {cleaned_preview}...", "DEBUG")

        if run_id:
            self._current_run_id = run_id

        if state == "delta":
            # New text arriving — cancel any pending deferred final
            # (direct gateway delta, not just agent stream).
            self._cancel_deferred_final()

            # Частичный ответ (streaming token)
            # ВАЖНО: разные провайдеры могут присылать delta в двух форматах:
            # 1) cumulative: content = весь накопленный текст
            # 2) incremental: content = только новый кусок
            # Делаем сборку, устойчивую к обоим форматам.
            if content:
                new_text = ""
                acquired = self._buffer_lock.acquire(timeout=5.0)
                if not acquired:
                    self._log_ws("CRITICAL: _buffer_lock stuck for 5s in delta handler, processing without lock", "ERROR")
                try:
                    prev_text = self._accumulated_text

                    # Формат 1: cumulative (новый content начинается с уже накопленного текста)
                    if content.startswith(self._accumulated_text):
                        new_text = content[len(self._accumulated_text):]
                        updated_text = content
                    # Если прислали старый/укороченный snapshot — не дублируем
                    elif self._accumulated_text.startswith(content):
                        new_text = ""
                        updated_text = self._accumulated_text
                    else:
                        # Формат 2: incremental или частичное рассинхронирование.
                        # Пробуем найти overlap (суффикс prev == префикс content),
                        # чтобы не дублировать символы при склейке.
                        overlap = 0
                        max_overlap = min(len(prev_text), len(content))
                        for i in range(max_overlap, 0, -1):
                            if prev_text.endswith(content[:i]):
                                overlap = i
                                break

                        if overlap > 0:
                            new_text = content[overlap:]
                        else:
                            # Нет overlap: считаем это чисто новым инкрементом
                            new_text = content

                        updated_text = prev_text + new_text

                    # Обновляем накопленный текст (append-friendly)
                    self._accumulated_text = updated_text
                    self._full_response = updated_text

                    # Логируем реальный инкремент и общий размер после сборки
                    self._log_ws(
                        f"Chat delta: +{len(new_text)} chars (cumulative: {len(updated_text)})",
                        "DEBUG"
                    )
                finally:
                    if acquired:
                        self._buffer_lock.release()

                # Callback OUTSIDE _buffer_lock to prevent lock starvation
                if new_text and self.on_token:
                    self.on_token(new_text)

        elif state == "final":
            if run_id:
                with self._buffer_lock:
                    if run_id in self._seen_final_run_ids:
                        self._log_ws(f"Duplicate final ignored for runId={run_id}", "DEBUG")
                        return
                    self._seen_final_run_ids.add(run_id)
                    if len(self._seen_final_run_ids) > 500:
                        self._seen_final_run_ids.clear()

            # Финальный полный ответ
            if content:
                with self._buffer_lock:
                    # final содержит полный текст — перезаписываем
                    self._full_response = content

                self._log_ws(f"Chat final: {len(content)} chars", "INFO")
            else:
                # FALLBACK: если final пришёл с пустым content, но есть накопленный текст от delta
                with self._buffer_lock:
                    if self._accumulated_text and not self._full_response:
                        self._full_response = self._accumulated_text
                        self._log_ws(f"Chat final (fallback): using accumulated {len(self._full_response)} chars", "WARN")
                    elif not self._full_response:
                        self._log_ws("Chat final: EMPTY content and no accumulated text!", "ERROR")

            # Вызываем on_complete только для финального ответа
            # Это остановит StreamingTTSManager и отправит остаток буфера
            if self._callback_mode == "streaming" and self.on_complete:
                self.on_complete(self._full_response)

            self._is_streaming = False
            self._is_processing = False
            self._response_event.set()

        elif state == "error":
            error_msg = payload.get("errorMessage", "Unknown chat error")
            self._log_ws(f"Chat error: {error_msg}", "ERROR")
            self._is_streaming = False
            self._is_processing = False

            # Сохраняем ошибку как ответ
            with self._buffer_lock:
                if not self._full_response:
                    self._full_response = f"Ошибка: {error_msg}"

            # Notify streaming completion so service.py stops TTS manager
            if self._callback_mode == "streaming" and self.on_complete:
                self.on_complete(self._full_response)

            self._response_event.set()

        elif state == "aborted":
            self._log_ws("Chat aborted", "WARN")
            self._is_streaming = False
            self._is_processing = False

            # Notify streaming completion
            if self._callback_mode == "streaming" and self.on_complete:
                self.on_complete(self._full_response)

            self._response_event.set()
        else:
            # Неизвестное состояние логируем явно, чтобы видеть формат gateway.
            # Treat any unrecognized state as terminal to prevent hanging forever
            # (watchdog would eventually time out, but this is faster).
            self._log_ws(
                f"Unknown chat state='{state}'. Treating as terminal. "
                f"Payload keys: {list(payload.keys())[:12]}",
                "WARN"
            )
            self._is_streaming = False
            self._is_processing = False

            if self._callback_mode == "streaming" and self.on_complete:
                with self._buffer_lock:
                    if not self._full_response and self._accumulated_text:
                        self._full_response = self._accumulated_text
                self.on_complete(self._full_response)

            self._response_event.set()

    def _send_connect(self, nonce: str):
        """Отправляет connect request с правильными ConnectParams (протокол v3).

        ВАЖНО: additionalProperties: false — нельзя добавлять лишние поля!
        """
        import platform

        req_id = str(uuid4())

        connect_params = {
            "minProtocol": self.PROTOCOL_VERSION,
            "maxProtocol": self.PROTOCOL_VERSION,
            "client": {
                "id": "gateway-client",
                "version": "1.0.0",
                "platform": "win32" if sys.platform == "win32" else sys.platform,
                "mode": "backend"
            },
            "role": "operator",
            "scopes": ["operator.admin"],
            "caps": [],
            "auth": {
                "token": self._gateway_token
            },
            "device": self._build_device_auth(nonce),
            "locale": "ru-RU",
            "userAgent": "kiwi-voice/1.0"
        }

        frame = {
            "type": "req",
            "id": req_id,
            "method": "connect",
            "params": connect_params
        }

        # Регистрируем pending request
        with self._pending_lock:
            self._pending_requests[req_id] = {
                "event": threading.Event(),
                "ok": None,
                "payload": None,
                "method": "connect"
            }

        try:
            self._ws.send(json.dumps(frame))
            self._log_ws(f"Connect request sent (id={req_id[:8]}...)", "INFO")
        except Exception as e:
            self._log_ws(f"Failed to send connect request: {e}", "ERROR")
            with self._pending_lock:
                self._pending_requests.pop(req_id, None)

    def _request(self, method: str, params: dict, timeout: float = 30.0) -> dict:
        """Отправляет request и ждёт response (блокирующий).

        Args:
            method: Имя метода (e.g. "chat.send", "chat.abort")
            params: Параметры запроса
            timeout: Таймаут ожидания ответа

        Returns:
            {"ok": bool, "payload": dict} или {"ok": False, "error": str}
        """
        if not self._is_authenticated:
            return {"ok": False, "error": "Not authenticated"}

        req_id = str(uuid4())

        frame = {
            "type": "req",
            "id": req_id,
            "method": method,
            "params": params
        }

        # Регистрируем pending request
        pending_event = threading.Event()
        with self._pending_lock:
            self._pending_requests[req_id] = {
                "event": pending_event,
                "ok": None,
                "payload": None,
                "method": method
            }

        try:
            self._ws.send(json.dumps(frame))
            self._log_ws(f"Request sent: {method} (id={req_id[:8]}...)", "DEBUG")
        except Exception as e:
            with self._pending_lock:
                self._pending_requests.pop(req_id, None)
            return {"ok": False, "error": f"Send failed: {e}"}

        # Ждём ответ
        if pending_event.wait(timeout=timeout):
            with self._pending_lock:
                result = self._pending_requests.pop(req_id, {})
            return {"ok": result.get("ok", False), "payload": result.get("payload", {})}
        else:
            with self._pending_lock:
                self._pending_requests.pop(req_id, None)
            return {"ok": False, "error": f"Timeout after {timeout}s"}

    def chat(self, message: str) -> str:

        """Отправляет сообщение и ожидает полный ответ (блокирующий вызов).

        Использует протокол Gateway v3: chat.send request + chat events.
        Совместим с интерфейсом OpenClawCLI.chat()

        Args:
            message: Сообщение для отправки

        Returns:
            Полный текст ответа или сообщение об ошибке
        """
        # Сбрасываем состояние
        with self._buffer_lock:
            self._full_response = ""
            self._accumulated_text = ""
        self._response_event.clear()
        self._is_processing = True
        self._is_streaming = True
        self._current_run_id = None

        # Отправляем сообщение через chat.send
        if not self.send_message(message, callback_mode="blocking"):
            self._is_processing = False
            self._is_streaming = False
            return "Ошибка: не удалось отправить сообщение через WebSocket"

        # Ждём завершения (final/error/aborted event) с таймаутом
        timeout = self.config.openclaw_timeout
        self._log_ws(f"Waiting for chat response (timeout={timeout}s)...", "DEBUG")

        if self._response_event.wait(timeout=timeout):
            response = self._full_response
            # Гарантируем строку
            if isinstance(response, list):
                response = "".join(str(r) for r in response)
            self._log_ws(f"Chat response received: {len(response)} chars", "INFO")
            return response if response else "Извини, я не получила ответ."
        else:
            self._log_ws(f"Chat response timeout after {timeout}s", "WARN")
            # Пытаемся отменить
            self.cancel()
            return "Извини, ответ занял слишком много времени."

    def send_message(
        self,
        message: str,
        context: Optional[str] = None,
        callback_mode: str = "streaming",
    ) -> bool:
        """Отправляет сообщение через WebSocket используя chat.send (протокол v3).

        Args:
            message: Сообщение для отправки
            context: Опциональный контекст (добавляется к сообщению)

        Returns:
            True если запрос отправлен успешно
        """
        if not self._is_authenticated:
            self._log_ws("Not authenticated, trying to connect...", "WARN")
            if not self.connect():
                return False

        # Сбрасываем состояние стриминга перед новым запросом
        self._cancel_deferred_final()  # new request overrides any pending final
        acquired = self._buffer_lock.acquire(timeout=3.0)
        if acquired:
            try:
                self._full_response = ""
                self._accumulated_text = ""
                self._seen_final_run_ids.clear()
            finally:
                self._buffer_lock.release()
        else:
            self._log_ws("send_message: _buffer_lock timeout (3s), clearing without lock", "WARNING")
            self._full_response = ""
            self._accumulated_text = ""
        self._callback_mode = callback_mode if callback_mode in ("streaming", "blocking") else "streaming"
        self._response_event.clear()
        self._current_run_id = None

        # Формируем полное сообщение с контекстом
        full_message = message
        if context:
            full_message = f"{context}\n{message}"

        try:
            # chat.send request по протоколу v3
            req_id = str(uuid4())
            idempotency_key = str(uuid4())

            chat_params = {
                "sessionKey": self._session_key,
                "message": full_message,
                "idempotencyKey": idempotency_key,
                "timeoutMs": self.config.openclaw_timeout * 1000
            }

            frame = {
                "type": "req",
                "id": req_id,
                "method": "chat.send",
                "params": chat_params
            }

            # Регистрируем pending request для chat.send response
            with self._pending_lock:
                self._pending_requests[req_id] = {
                    "event": threading.Event(),
                    "ok": None,
                    "payload": None,
                    "method": "chat.send"
                }

            self._ws.send(json.dumps(frame))
            self._is_streaming = True
            self._is_processing = True

            self._log_ws(f"chat.send sent (id={req_id[:8]}..., session={self._session_key}): {message[:60]}...", "INFO")
            return True

        except Exception as e:
            self._log_ws(f"chat.send error: {e}", "ERROR")
            self._is_connected = False
            self._is_authenticated = False
            self._is_processing = False
            return False

    def is_processing(self) -> bool:
        """Проверяет, выполняется ли сейчас обработка запроса."""
        return self._is_processing

    def is_streaming(self) -> bool:
        """Возвращает True, если сейчас идёт streaming ответ."""
        return self._is_streaming

    def cancel(self) -> bool:
        """Прерывает текущую обработку через chat.abort (протокол v3)."""
        self._cancel_deferred_final()  # cancel overrides pending final
        self._log_ws("Cancel requested (chat.abort)", "WARN")

        if self._is_authenticated and (self._is_processing or self._is_streaming):
            try:
                abort_params = {
                    "sessionKey": self._session_key,
                }
                # Добавляем runId если известен
                if self._current_run_id:
                    abort_params["runId"] = self._current_run_id

                # Отправляем abort асинхронно (не ждём ответ)
                req_id = str(uuid4())
                frame = {
                    "type": "req",
                    "id": req_id,
                    "method": "chat.abort",
                    "params": abort_params
                }

                # Регистрируем pending (но не блокируем)
                with self._pending_lock:
                    self._pending_requests[req_id] = {
                        "event": threading.Event(),
                        "ok": None,
                        "payload": None,
                        "method": "chat.abort"
                    }

                self._ws.send(json.dumps(frame))
                self._log_ws(f"chat.abort sent (runId={self._current_run_id or 'none'})", "INFO")
            except Exception as e:
                self._log_ws(f"chat.abort error: {e}", "ERROR")

        # В любом случае сбрасываем состояние
        self._is_streaming = False
        self._is_processing = False
        self._current_run_id = None
        acquired = self._buffer_lock.acquire(timeout=3.0)
        if acquired:
            try:
                self._accumulated_text = ""
                self._full_response = ""
            finally:
                self._buffer_lock.release()
        else:
            self._log_ws("cancel: _buffer_lock timeout (3s), clearing without lock", "WARNING")
            self._accumulated_text = ""
            self._full_response = ""
        self._response_event.set()
        return True

    def close(self):
        """Закрывает WebSocket соединение."""
        self._cancel_deferred_final()
        self._log_ws("Closing connection...", "INFO")
        self._stop_event.set()

        if self._ws:
            try:
                self._ws.close()
            except Exception as e:
                self._log_ws(f"Error closing WebSocket: {e}", "DEBUG")

        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)

        self._is_connected = False
        self._is_streaming = False
        self._is_processing = False
        self._log_ws("Connection closed", "INFO")
