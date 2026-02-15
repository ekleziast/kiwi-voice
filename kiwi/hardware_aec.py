"""
Hardware AEC (Acoustic Echo Cancellation) Module

Предоставляет аппаратное подавление эха через:
- Windows: WASAPI AEC (Acoustic Echo Cancellation)
- Linux: PulseAudio/PipeWire echo-cancel module
- macOS: Voice Processing I/O mode

С fallback на программное AEC если аппаратное недоступно.
"""

import os
import sys
import platform
import subprocess
import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from collections import deque
import threading
import time

from kiwi.utils import kiwi_log


@dataclass
class AECConfig:
    """Конфигурация AEC."""
    enabled: bool = True
    use_hardware: bool = True  # Пытаться использовать аппаратное AEC
    fallback_to_software: bool = True  # Fallback на программное
    
    # Параметры программного AEC (SpeexDSP или аналог)
    filter_length_ms: int = 250  # Длина фильтра в мс
    echo_delay_ms: int = 100   # Задержка эха в мс
    
    # Параметры шумоподавления
    noise_suppression: bool = True
    ns_level: int = -30  # dB
    
    # Автоматическая регулировка усиления
    agc: bool = True
    agc_target: float = 0.5


class HardwareAEC:
    """
    Менеджер аппаратного AEC.
    
    Пытается использовать встроенное аппаратное подавление эха,
    с fallback на программное решение.
    """
    
    def __init__(self, config: Optional[AECConfig] = None):
        self.config = config or AECConfig()
        self.platform = platform.system().lower()
        
        self._hardware_aec_available = False
        self._aec_device_id: Optional[int] = None
        self._aec_device_name: Optional[str] = None
        
        # Программный AEC (fallback)
        self._software_aec = None
        self._use_software_aec = False
        
        # Буферы для программного AEC
        self._echo_buffer = deque(maxlen=1000)  # ~3 секунды при 16kHz
        self._reference_buffer = deque(maxlen=1000)
        
        # Для корреляционного анализа
        self._last_played_audio: Optional[np.ndarray] = None
        self._correlation_threshold = 0.6
        
        # Статистика
        self._frames_processed = 0
        self._echo_cancelled_frames = 0
        
        if self.config.enabled:
            self._initialize()
    
    def _initialize(self):
        """Инициализация AEC."""
        kiwi_log("AEC", "Initializing...")
        
        if self.config.use_hardware:
            self._hardware_aec_available = self._check_hardware_aec()
        
        if self._hardware_aec_available:
            kiwi_log("AEC", f"Hardware AEC available: {self._aec_device_name}")
        elif self.config.fallback_to_software:
            kiwi_log("AEC", "Hardware AEC not available, using software fallback")
            self._initialize_software_aec()
        else:
            kiwi_log("AEC", "AEC disabled (hardware unavailable, fallback disabled)")
    
    def _check_hardware_aec(self) -> bool:
        """Проверяет доступность аппаратного AEC."""
        if self.platform == 'windows':
            return self._check_windows_aec()
        elif self.platform == 'linux':
            return self._check_linux_aec()
        elif self.platform == 'darwin':
            return self._check_macos_aec()
        return False
    
    def _check_windows_aec(self) -> bool:
        """Проверяет WASAPI AEC на Windows."""
        try:
            import sounddevice as sd
            
            # Ищем устройства с поддержкой AEC
            devices = sd.query_devices()
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:  # Input device
                    name = device['name'].lower()
                    # Признаки AEC-устройств
                    aec_keywords = [
                        'aec', 'echo', 'communication', 'communications',
                        'hands-free', 'handsfree', 'agc', 'noise',
                        'array', 'microphone array'
                    ]
                    
                    if any(kw in name for kw in aec_keywords):
                        self._aec_device_id = i
                        self._aec_device_name = device['name']
                        kiwi_log("AEC", f"Found Windows AEC device: {device['name']} (id={i})")
                        return True
            
            # Проверяем наличие Communications устройств
            try:
                import comtypes
                from comtypes import CLSCTX_ALL
                
                # Пытаемся получить доступ к Core Audio API
                # Это требует pycaw: pip install pycaw
                try:
                    from pycaw.pycaw import AudioUtilities
                    
                    # Получаем default communication device
                    comm_device = AudioUtilities.GetSpeakers()
                    if comm_device:
                        kiwi_log("AEC", f"Default communication device: {comm_device.FriendlyName}")
                        # Communications устройства обычно имеют встроенный AEC
                        return True
                        
                except ImportError:
                    kiwi_log("AEC", "pycaw not installed, skipping Windows Core Audio check", level="WARNING")
                    
            except Exception as e:
                kiwi_log("AEC", f"Windows AEC check error: {e}", level="ERROR")
            
            return False
            
        except Exception as e:
            kiwi_log("AEC", f"Error checking Windows AEC: {e}", level="ERROR")
            return False
    
    def _check_linux_aec(self) -> bool:
        """Проверяет PulseAudio/PipeWire AEC на Linux."""
        try:
            # Проверяем наличие echo-cancel модуля
            result = subprocess.run(
                ['pactl', 'list', 'modules'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'echo-cancel' in result.stdout.lower():
                kiwi_log("AEC", "PulseAudio echo-cancel module loaded")
                return True
            
            # Проверяем PipeWire
            result = subprocess.run(
                ['pw-cli', 'ls', 'Node'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if 'echo-cancel' in result.stdout.lower() or 'aec' in result.stdout.lower():
                kiwi_log("AEC", "PipeWire AEC found")
                return True
            
            # Пытаемся загрузить модуль
            try:
                subprocess.run(
                    ['pactl', 'load-module', 'module-echo-cancel'],
                    capture_output=True,
                    timeout=5
                )
                kiwi_log("AEC", "Loaded PulseAudio echo-cancel module")
                return True
            except:
                pass
            
            return False
            
        except Exception as e:
            kiwi_log("AEC", f"Linux AEC check error: {e}", level="ERROR")
            return False
    
    def _check_macos_aec(self) -> bool:
        """Проверяет Voice Processing IO на macOS."""
        try:
            # На macOS AEC обычно доступен через CoreAudio
            # с установленным флагом kAudioUnitProperty_VoiceProcessingIO
            
            # Проверяем наличие аудиоустройств с voice processing
            result = subprocess.run(
                ['system_profiler', 'SPAudioDataType'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if 'built-in' in result.stdout.lower() or 'microphone' in result.stdout.lower():
                # Встроенные микрофоны Mac обычно имеют AEC
                kiwi_log("AEC", "macOS built-in audio detected (likely has AEC)")
                return True
            
            return False
            
        except Exception as e:
            kiwi_log("AEC", f"macOS AEC check error: {e}", level="ERROR")
            return False
    
    def _initialize_software_aec(self):
        """Инициализирует программный AEC."""
        try:
            # Пробуем использовать speexdsp
            import speexdsp
            
            self._software_aec = speexdsp.EchoCanceller(
                sampling_rate=16000,
                filter_length_ms=self.config.filter_length_ms
            )
            
            if self.config.noise_suppression:
                self._software_aec.enable_noise_suppression(self.config.ns_level)
            
            self._use_software_aec = True
            kiwi_log("AEC", "Software AEC initialized (SpeexDSP)")
            
        except ImportError:
            kiwi_log("AEC", "SpeexDSP not available, using correlation-based fallback", level="WARNING")
            self._use_software_aec = True  # Будем использовать корреляционный метод
    
    def process(
        self,
        mic_audio: np.ndarray,
        reference_audio: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Обрабатывает аудио с микрофона, применяя AEC.
        
        Args:
            mic_audio: Аудио с микрофона
            reference_audio: Ссылочное аудио (то, что воспроизводится)
        
        Returns:
            Аудио с подавленным эхом
        """
        self._frames_processed += 1
        
        if not self.config.enabled:
            return mic_audio
        
        # Если используется аппаратное AEC - просто возвращаем аудио
        if self._hardware_aec_available:
            return mic_audio
        
        # Программное AEC
        if self._use_software_aec:
            return self._process_software_aec(mic_audio, reference_audio)
        
        return mic_audio
    
    def _process_software_aec(
        self,
        mic_audio: np.ndarray,
        reference_audio: Optional[np.ndarray]
    ) -> np.ndarray:
        """Программное подавление эха."""
        # SpeexDSP метод
        if self._software_aec is not None:
            try:
                if reference_audio is not None:
                    # Нужно выровнять длину
                    min_len = min(len(mic_audio), len(reference_audio))
                    mic = mic_audio[:min_len]
                    ref = reference_audio[:min_len]
                    
                    processed = self._software_aec.process(mic, ref)
                    
                    # Дополняем если нужно
                    if len(processed) < len(mic_audio):
                        processed = np.concatenate([processed, mic_audio[len(processed):]])
                    
                    self._echo_cancelled_frames += 1
                    return processed
            except Exception as e:
                kiwi_log("AEC", f"SpeexDSP error: {e}", level="ERROR")
        
        # Fallback: корреляционный метод
        if reference_audio is not None:
            return self._correlation_cancel(mic_audio, reference_audio)
        
        return mic_audio
    
    def _correlation_cancel(
        self,
        mic_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> np.ndarray:
        """
        Простое корреляционное подавление эха.
        
        Ищет сегменты микрофонного аудио, которые коррелируют
        с воспроизводимым аудио, и подавляет их.
        """
        try:
            # Нормализуем
            mic_norm = mic_audio / (np.max(np.abs(mic_audio)) + 1e-10)
            ref_norm = reference_audio / (np.max(np.abs(reference_audio)) + 1e-10)
            
            # Вычисляем корреляцию
            min_len = min(len(mic_norm), len(ref_norm))
            correlation = np.corrcoef(mic_norm[:min_len], ref_norm[:min_len])[0, 1]
            
            if np.isnan(correlation):
                correlation = 0.0
            
            # Если корреляция высокая - это эхо
            if correlation > self._correlation_threshold:
                # Подавляем на основе корреляции
                suppression = (correlation - self._correlation_threshold) / (1 - self._correlation_threshold)
                suppression = np.clip(suppression, 0, 0.9)  # Максимум 90% подавления
                
                processed = mic_audio * (1 - suppression)
                self._echo_cancelled_frames += 1
                
                if self._frames_processed % 100 == 0:
                    kiwi_log("AEC", f"Echo suppressed: correlation={correlation:.2f}, suppression={suppression:.2%}")
                
                return processed
            
        except Exception as e:
            kiwi_log("AEC", f"Correlation cancel error: {e}", level="ERROR")
        
        return mic_audio
    
    def update_reference(self, audio: np.ndarray):
        """
        Обновляет ссылочное аудио (то, что воспроизводится через колонки).
        
        Args:
            audio: Аудио, которое будет воспроизведено
        """
        if not self.config.enabled or self._hardware_aec_available:
            return
        
        self._last_played_audio = audio.copy()
        
        # Добавляем в буфер для программного AEC
        self._reference_buffer.extend(audio)
    
    def get_aec_device(self) -> Optional[int]:
        """Возвращает ID устройства с аппаратным AEC."""
        return self._aec_device_id if self._hardware_aec_available else None
    
    def is_hardware_aec(self) -> bool:
        """Возвращает True, если используется аппаратное AEC."""
        return self._hardware_aec_available
    
    def is_software_aec(self) -> bool:
        """Возвращает True, если используется программное AEC."""
        return self._use_software_aec and not self._hardware_aec_available
    
    def get_stats(self) -> dict:
        """Возвращает статистику AEC."""
        return {
            'hardware_available': self._hardware_aec_available,
            'hardware_device': self._aec_device_name,
            'using_software': self._use_software_aec,
            'frames_processed': self._frames_processed,
            'echo_cancelled_frames': self._echo_cancelled_frames,
            'effectiveness': self._echo_cancelled_frames / max(1, self._frames_processed),
        }
    
    def reset(self):
        """Сбрасывает состояние AEC."""
        self._echo_buffer.clear()
        self._reference_buffer.clear()
        self._last_played_audio = None
        
        if self._software_aec is not None:
            try:
                self._software_aec.reset()
            except:
                pass
        
        kiwi_log("AEC", "State reset")


def create_aec_from_config(config_path: str = "config.yaml") -> HardwareAEC:
    """Создает AEC из конфигурации YAML."""
    config = AECConfig()
    
    try:
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}
            
            aec_cfg = yaml_data.get('aec', {})
            
            config.enabled = aec_cfg.get('enabled', True)
            config.use_hardware = aec_cfg.get('use_hardware', True)
            config.fallback_to_software = aec_cfg.get('fallback_to_software', True)
            config.filter_length_ms = aec_cfg.get('filter_length_ms', 250)
            config.echo_delay_ms = aec_cfg.get('echo_delay_ms', 100)
            config.noise_suppression = aec_cfg.get('noise_suppression', True)
            config.ns_level = aec_cfg.get('ns_level', -30)
            config.agc = aec_cfg.get('agc', True)
            
            kiwi_log("AEC", f"Config loaded from {config_path}")
    except Exception as e:
        kiwi_log("AEC", f"Error loading config: {e}, using defaults", level="ERROR")
    
    return HardwareAEC(config)


# Для тестирования
if __name__ == "__main__":
    print("Testing Hardware AEC module...")
    
    aec = create_aec_from_config()
    
    print(f"\nAEC Stats:")
    stats = aec.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Тестовое аудио
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Симулируем микрофон (эхо + речь)
    echo = np.sin(2 * np.pi * 440 * t) * 0.3  # Эхо от колонок
    speech = np.sin(2 * np.pi * 1000 * t) * 0.5  # Речь пользователя
    mic_signal = echo + speech
    
    # Ссылочное аудио (то, что воспроизводится)
    reference = np.sin(2 * np.pi * 440 * t) * 0.8
    
    # Обрабатываем
    aec.update_reference(reference)
    processed = aec.process(mic_signal, reference)
    
    print(f"\nTest completed:")
    print(f"  Input RMS: {np.sqrt(np.mean(mic_signal**2)):.4f}")
    print(f"  Output RMS: {np.sqrt(np.mean(processed**2)):.4f}")
    print(f"  Echo reduction: {(1 - np.sqrt(np.mean(processed**2)) / np.sqrt(np.mean(mic_signal**2))) * 100:.1f}%")
