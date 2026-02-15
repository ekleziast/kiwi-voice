#!/usr/bin/env python3
"""OpenClaw CLI client for Kiwi Voice."""

import os
import re
import subprocess
import sys
import time
from typing import Optional

from kiwi.utils import kiwi_log


class OpenClawCLI:
    """Клиент для общения с OpenClaw через CLI."""

    def __init__(
        self,
        openclaw_bin: str = "openclaw",
        session_id: str = "kiwi-voice",
        agent: Optional[str] = None,
        timeout: int = 120,
        model: Optional[str] = None,
        retry_max: int = 3,
        retry_delays: list = None,
    ):
        self.openclaw_bin = self._resolve_openclaw_path(openclaw_bin)
        self.session_id = session_id
        self.agent = agent
        self.timeout = timeout
        self.model = model
        self.retry_max = retry_max
        self.retry_delays = retry_delays or [0.5, 1.0, 2.0]
        self.session_key = f"agent:{self.session_id}:{self.session_id}"
        self._current_process: Optional[subprocess.Popen] = None
        self._is_processing = False
        self._check_cli()

    def _resolve_openclaw_path(self, openclaw_bin: str) -> str:
        """Returns bin as-is, relying on PATH unless an explicit file path is provided."""
        if os.path.exists(openclaw_bin):
            return openclaw_bin
        return openclaw_bin

    def _get_command(self, args: list) -> list:
        """Формирует команду с учётом платформы."""
        if self.openclaw_bin.endswith('.mjs'):
            return ["node", self.openclaw_bin] + args
        return [self.openclaw_bin] + args

    def _check_cli(self):
        """Проверяет доступность openclaw CLI."""
        try:
            cmd = self._get_command(["--version"])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                kiwi_log("OPENCLAW", f"CLI found: {version}", level="INFO")
                if self.openclaw_bin.endswith('.mjs'):
                    kiwi_log("OPENCLAW", f"Using: node {self.openclaw_bin}", level="INFO")
            else:
                kiwi_log("OPENCLAW", f"CLI check failed: {result.stderr}", level="ERROR")
        except FileNotFoundError:
            kiwi_log("OPENCLAW", f"ERROR: '{self.openclaw_bin}' not found", level="ERROR")
            kiwi_log("OPENCLAW", "Make sure OpenClaw is installed: npm install -g openclaw", level="ERROR")
            sys.exit(1)
        except Exception as e:
            kiwi_log("OPENCLAW", f"CLI check error: {e}", level="ERROR")

    def is_processing(self) -> bool:
        """Проверяет, выполняется ли сейчас обработка."""
        return self._is_processing

    def cancel(self) -> bool:
        """Прерывает текущую обработку."""
        if self._current_process and self._is_processing:
            kiwi_log("OPENCLAW", "Cancelling current operation...", level="INFO")
            try:
                self._current_process.terminate()
                self._current_process.wait(timeout=2)
                self._is_processing = False
                self._current_process = None
                kiwi_log("OPENCLAW", "Cancelled successfully", level="INFO")
                return True
            except Exception as e:
                kiwi_log("OPENCLAW", f"Cancel error: {e}", level="ERROR")
                try:
                    self._current_process.kill()
                except:
                    pass
                self._is_processing = False
                self._current_process = None
        return False

    def _is_rate_limit_error(self, stderr: str) -> bool:
        """Проверяет, является ли ошибка rate_limit."""
        if not stderr:
            return False
        rate_limit_indicators = [
            "rate_limit",
            "rate limit",
            "cooldown",
            "all profiles unavailable",
            "Provider openrouter is in cooldown",
        ]
        stderr_lower = stderr.lower()
        return any(indicator in stderr_lower for indicator in rate_limit_indicators)

    def chat(self, message: str) -> str:
        """Отправляет сообщение в существующую сессию через agent CLI с retry при rate_limit.

        ИСПРАВЛЕНО: Использует subprocess.run() вместо ненадёжного стримингового чтения.
        """
        args = [
            "agent",
            "--session-id", self.session_id,
            "--message", message,
            "--timeout", str(self.timeout),
        ]

        if self.agent:
            args.extend(["--agent", self.agent])

        cmd = self._get_command(args)

        # Retry loop с нарастающими задержками
        for attempt in range(self.retry_max + 1):
            if attempt > 0:
                delay = self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                kiwi_log("OPENCLAW", f"Retry {attempt}/{self.retry_max} after {delay}s...", level="WARNING")
                time.sleep(delay)

            kiwi_log("OPENCLAW", f"Sending to session {self.session_id}: {message[:50]}..." + (f" (attempt {attempt + 1})" if attempt > 0 else ""), level="INFO")
            self._is_processing = True

            try:
                # ИСПРАВЛЕНИЕ: Используем subprocess.run() вместо Popen + стриминг
                # Это надёжнее и гарантирует чтение всего stdout
                start_time = time.time()
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=self.timeout + 30,
                )

                stdout = result.stdout
                stderr = result.stderr
                returncode = result.returncode
                self._is_processing = False

                if returncode == 0:
                    response = self._clean_response(stdout)

                    if response:
                        total_time = time.time() - start_time
                        kiwi_log("OPENCLAW", f"Response complete ({total_time:.2f}s): {response[:80]}...", level="INFO")
                        return response
                    else:
                        kiwi_log("OPENCLAW", "Empty response after cleaning", level="WARNING")
                        return "Извини, я не получила ответ."
                else:
                    # Проверяем, является ли ошибка rate_limit
                    if self._is_rate_limit_error(stderr) and attempt < self.retry_max:
                        kiwi_log("OPENCLAW", "Rate limit detected, will retry...", level="WARNING")
                        continue

                    kiwi_log("OPENCLAW", f"CLI error (code {returncode})", level="ERROR")
                    kiwi_log("OPENCLAW", f"stderr: {stderr[:200]}", level="ERROR")
                    return "Извини, произошла ошибка при обработке запроса."

            except subprocess.TimeoutExpired:
                self._is_processing = False
                kiwi_log("OPENCLAW", "Timeout expired", level="WARNING")
                return "Извини, ответ занял слишком много времени."
            except Exception as e:
                self._is_processing = False
                kiwi_log("OPENCLAW", f"Error: {e}", level="ERROR")
                return f"Ошибка: {str(e)}"

        # Все попытки исчерпаны
        return "Извини, сервис временно недоступен (rate limit). Попробуй позже."

    def _clean_response(self, text: str) -> str:
        """Очищает ответ от баннера OpenClaw и лишнего форматирования."""
        if not text:
            return ""

        lines = text.split('\n')
        cleaned_lines = []

        # Паттерны для фильтрации баннера OpenClaw
        banner_patterns = [
            r'^🦞\s*OpenClaw',           # 🦞 OpenClaw ...
            r'^OpenClaw\s+\d',           # OpenClaw 2026.2.3...
            r'^\s*\|+\s*$',              # Спиннеры: |, ||, |||
            r'^\s*[o\-/\⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]+\s*$',  # Спиннеры анимации
            r'^\s*Your inbox.*',          # Баннер текст
            r'^\s*WhatsApp automation.*', # Баннер текст
            r'^\s*EXFOLIATE.*',           # Баннер текст
        ]

        for line in lines:
            line_stripped = line.strip()

            # Пропускаем пустые строки
            if not line_stripped:
                continue

            # Проверяем паттерны баннера
            is_banner = False
            for pattern in banner_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    is_banner = True
                    break

            if is_banner:
                continue

            # Ищем строку с ответом Киви (начинается с 🥝)
            if line_stripped.startswith('🥝'):
                # Извлекаем текст после эмодзи и пробелов
                response_text = line_stripped[1:].strip()
                if response_text:
                    cleaned_lines.append(response_text)
            else:
                cleaned_lines.append(line_stripped)

        # Объединяем строки
        text = ' '.join(cleaned_lines).strip()

        # Убираем markdown форматирование
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'#+\s*', '', text)

        # Убираем лишние пробелы
        while '  ' in text:
            text = text.replace('  ', ' ')

        # Убираем начальное "Киви, " или "Киви " из ответа
        text_lower = text.lower()
        if text_lower.startswith('киви, '):
            text = text[6:].strip()
            kiwi_log("CLEAN", "Removed 'Киви, ' prefix from response", level="INFO")
        elif text_lower.startswith('киви '):
            text = text[5:].strip()
            kiwi_log("CLEAN", "Removed 'Киви ' prefix from response", level="INFO")

        return text
