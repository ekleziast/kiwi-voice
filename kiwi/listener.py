#!/usr/bin/env python3
"""
Kiwi Voice Listener - Faster Whisper + Wake Word Detection

РќРµРїСЂРµСЂС‹РІРЅРѕРµ РїСЂРѕСЃР»СѓС€РёРІР°РЅРёРµ РјРёРєСЂРѕС„РѕРЅР° СЃ СЂР°СЃРїРѕР·РЅР°РІР°РЅРёРµРј СЂРµС‡Рё.
Р РµР°РіРёСЂСѓРµС‚ С‚РѕР»СЊРєРѕ РЅР° РїСЂСЏРјРѕРµ РѕР±СЂР°С‰РµРЅРёРµ "РљРёРІРё, ...".

РћРїС‚РёРјРёР·Р°С†РёРё РґР»СЏ realtime:
- РђРґР°РїС‚РёРІРЅС‹Р№ РїРѕСЂРѕРі С€СѓРјР° (РєР°Р»РёР±СЂРѕРІРєР° РїСЂРё СЃС‚Р°СЂС‚Рµ)
- VAD (Voice Activity Detection) С‡РµСЂРµР· Whisper
- РњРёРЅРёРјР°Р»СЊРЅС‹Рµ Р·Р°РґРµСЂР¶РєРё РґР»СЏ Р±С‹СЃС‚СЂРѕРіРѕ РѕС‚РєР»РёРєР°
"""

import os
import re
import sys
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", module="pyannote")
warnings.filterwarnings("ignore", module="lightning")
warnings.filterwarnings("ignore", module="asteroid_filterbanks")
import queue
import threading
import time
import difflib
import traceback
from collections import deque
from typing import Optional, Callable, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Import logging utilities
try:
    from kiwi.utils import kiwi_log, log_crash
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    def kiwi_log(tag: str, message: str, level: str = "INFO"):
        print(f"[{tag}] {message}", flush=True)
    print("[WARN] utils.py not found, using basic logging")

# Unified VAD Pipeline
try:
    from kiwi.unified_vad import UnifiedVAD, VADResult
    UNIFIED_VAD_AVAILABLE = True
except ImportError:
    UNIFIED_VAD_AVAILABLE = False
    kiwi_log("LISTENER", "Unified VAD not available")

# Noise Reduction (spectral gating)
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    kiwi_log("LISTENER", "noisereduce not available — noise suppression disabled")

# Torch for Silero VAD (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    kiwi_log("LISTENER", "torch not available — Silero VAD disabled")

# Speaker Identification (Phase 1: echo cancellation)
try:
    from kiwi.speaker_id import SpeakerIdentifier
    SPEAKER_ID_AVAILABLE = True
except ImportError:
    SPEAKER_ID_AVAILABLE = False
    kiwi_log("LISTENER", "Speaker identification not available")

# Speaker Manager (Priority + Access Control)
try:
    from kiwi.speaker_manager import SpeakerManager, VoicePriority
    SPEAKER_MANAGER_AVAILABLE = True
except ImportError:
    SPEAKER_MANAGER_AVAILABLE = False
    kiwi_log("LISTENER", "Speaker Manager not available")

# Voice Security (Dangerous commands + Telegram approval)
try:
    from kiwi.voice_security import VoiceSecurity, OWNER_CONTROL_PATTERNS, extract_name_from_command
    VOICE_SECURITY_AVAILABLE = True
except ImportError:
    VOICE_SECURITY_AVAILABLE = False
    kiwi_log("LISTENER", "Voice Security not available")

# РљРѕРЅС„РёРіСѓСЂР°С†РёСЏ
WAKE_WORD = "РєРёРІРё"
POSITION_LIMIT = 5
FUZZY_MATCH_THRESHOLD = 0.60
SAMPLE_RATE = 16000

# РЎР»РѕРІР°, РєРѕС‚РѕСЂС‹Рµ РќР• РґРѕР»Р¶РЅС‹ РјР°С‚С‡РёС‚СЊСЃСЏ СЃ wake word С‡РµСЂРµР· fuzzy matching
# (С‡Р°СЃС‚С‹Рµ Р»РѕР¶РЅС‹Рµ СЃСЂР°Р±Р°С‚С‹РІР°РЅРёСЏ РЅР° РєРѕСЂРѕС‚РєРёРµ СЃР»РѕРІР° СЃ РѕР±С‰РёРјРё Р±СѓРєРІР°РјРё)
FUZZY_BLACKLIST = {"РёРґРё", "РёР»Рё", "РЅРё", "РєРё", "РёРІ", "С‚Рє", "РІС‹", "РјС‹", "РѕРЅ", "РѕРЅРѕ"}

# === РђР’РўРћРРЎРџР РђР’Р›Р•РќРР• РўР РђРќРЎРљР РРџР¦РР ===
# Р§Р°СЃС‚С‹Рµ РѕРїРµС‡Р°С‚РєРё wake word (С‚РѕР»СЊРєРѕ С†РµР»С‹Рµ СЃР»РѕРІР°!)
WAKE_WORD_TYPOS = {
    "РєРёРµРІРµ": "РєРёРІРё",
    "РєРёРµРІ": "РєРёРІРё",
    "РєРёРІРµРЅ": "РєРёРІРё",
    "РєРёРІС‹": "РєРёРІРё",
    "РєРёРІР°": "РєРёРІРё",
    "РєРёРІРёРё": "РєРёРІРё",
    "РєРёРІРёР№": "РєРёРІРё",
    "РєРёРІ": "РєРёРІРё",
    "РёРІРё": "РєРёРІРё",  # Whisper С‡Р°СЃС‚Рѕ РѕР±СЂРµР·Р°РµС‚ РїРµСЂРІСѓСЋ Р±СѓРєРІСѓ
    "РєРІРё": "РєРёРІРё",
    "С‚РёРІРё": "РєРёРІРё",
    "С‚РёРІРё,": "РєРёРІРё,",
    "С‚РёРІРё.": "РєРёРІРё.",
    # РќР• РґРѕР±Р°РІР»СЏС‚СЊ РєРѕСЂРѕС‚РєРёРµ: "РёРІ", "РєРё" - Р»РѕРјР°СЋС‚ РґСЂСѓРіРёРµ СЃР»РѕРІР°
}

# Whisper initial prompt вЂ” С‚РѕР»СЊРєРѕ wake word, Р±РµР· РєРѕРјР°РЅРґ (С‡С‚РѕР±С‹ РЅРµ РіР°Р»Р»СЋС†РёРЅРёСЂРѕРІР°С‚СЊ)
WHISPER_INITIAL_PROMPT = "РљРёРІРё"

# === KNOWN WHISPER HALLUCINATIONS (Russian) ===
# Whisper often generates these phrases from silence/noise/short audio.
WHISPER_HALLUCINATION_PATTERNS = [
    "редактор субтитров",
    "субтитры сделал",
    "субтитры подготовил",
    "продолжение следует",
    "подписывайтесь на канал",
    "спасибо за просмотр",
    "спасибо за подписку",
    "спасибо за внимание",
    "благодарю за внимание",
    "не забудьте подписаться",
    "ставьте лайк",
    "корректор",
    "переводчик",
    "звукорежиссёр",
    "звукорежиссер",
    "монтажёр",
    "монтажер",
    "автор сценария",
    "режиссёр монтажа",
    "режиссер монтажа",
    "srt subs",
    "amara.org",
]

# === REALTIME РћРџРўРРњРР—РђР¦РР ===
# РњРёРЅРёРјР°Р»СЊРЅР°СЏ РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ С‡Р°РЅРєР° РґР»СЏ Р±С‹СЃС‚СЂРѕР№ СЂРµР°РєС†РёРё
CHUNK_DURATION = 0.3  # Р‘С‹Р»Рѕ 0.5

# Pre-buffer: С…СЂР°РЅРёРј Р°СѓРґРёРѕ Р”Рћ РѕР±РЅР°СЂСѓР¶РµРЅРёСЏ СЂРµС‡Рё (С‡С‚РѕР±С‹ РЅРµ С‚РµСЂСЏС‚СЊ РЅР°С‡Р°Р»Рѕ С„СЂР°Р·С‹)
PRE_BUFFER_DURATION = 0.6  # 600ms РґРѕ РЅР°С‡Р°Р»Р° СЂРµС‡Рё (С…РІР°С‚РёС‚ РЅР° "РљРёРІРё")

# РђРґР°РїС‚РёРІРЅС‹Р№ РїРѕСЂРѕРі С€СѓРјР° (РєР°Р»РёР±СЂСѓРµС‚СЃСЏ РїСЂРё СЃС‚Р°СЂС‚Рµ)
NOISE_SAMPLE_DURATION = 2.0  # РЎРµРєСѓРЅРґ РґР»СЏ РєР°Р»РёР±СЂРѕРІРєРё С€СѓРјР°
NOISE_THRESHOLD_MULTIPLIER = 1.5  # РњРЅРѕР¶РёС‚РµР»СЊ РЅР°Рґ СѓСЂРѕРІРЅРµРј С€СѓРјР°

# РњРёРЅРёРјР°Р»СЊРЅР°СЏ РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ СЂРµС‡Рё РґР»СЏ СЂР°СЃРїРѕР·РЅР°РІР°РЅРёСЏ
MIN_SPEECH_DURATION = 0.3  # Р‘С‹Р»Рѕ 1.0 - СЂРµР°РіРёСЂСѓРµРј РЅР° РєРѕСЂРѕС‚РєРёРµ РєРѕРјР°РЅРґС‹

# РњР°РєСЃРёРјР°Р»СЊРЅР°СЏ РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ СЂРµС‡Рё (Р°РІР°СЂРёР№РЅС‹Р№ РїСЂРµРґРѕС…СЂР°РЅРёС‚РµР»СЊ вЂ” Р·Р°С‰РёС‚Р° РѕС‚ Р·Р°РІРёСЃР°РЅРёСЏ РїСЂРё РїРѕСЃС‚РѕСЏРЅРЅРѕРј С€СѓРјРµ)
# РЈРІРµР»РёС‡РµРЅРѕ СЃ 5.0 РґРѕ 30.0 СЃРµРєСѓРЅРґ вЂ” РїРѕР»СЊР·РѕРІР°С‚РµР»СЊ РґРѕР»Р¶РµРЅ СѓСЃРїРµС‚СЊ РґРѕРіРѕРІРѕСЂРёС‚СЊ РґР»РёРЅРЅСѓСЋ С„СЂР°Р·Сѓ
MAX_SPEECH_DURATION = 30.0

# Р”Р»РёС‚РµР»СЊРЅРѕСЃС‚СЊ С‚РёС€РёРЅС‹ РґР»СЏ Р·Р°РІРµСЂС€РµРЅРёСЏ utterance
# РЈРІРµР»РёС‡РµРЅРѕ СЃ 0.8 РґРѕ 1.5 СЃРµРєСѓРЅРґ вЂ” РґР°С‘С‚ РІСЂРµРјСЏ РЅР° РµСЃС‚РµСЃС‚РІРµРЅРЅС‹Рµ РїР°СѓР·С‹ РІРЅСѓС‚СЂРё С„СЂР°Р·С‹
SILENCE_DURATION_END = 1.5

# === Р—РђР©РРўРђ РћРў Р¤РђРќРўРћРњРќР«РҐ Р—Р’РЈРљРћР’ ===
# РњРёРЅРёРјР°Р»СЊРЅР°СЏ РіСЂРѕРјРєРѕСЃС‚СЊ РґР»СЏ РЅР°С‡Р°Р»Р° Р·Р°РїРёСЃРё (РѕС‚СЃРµРёРІР°РµС‚ С‚РёС…РёРµ С„РѕРЅРѕРІС‹Рµ Р·РІСѓРєРё)
MIN_SPEECH_VOLUME = 0.015  # РњРёРЅРёРјСѓРј 1.5% РѕС‚ РјР°РєСЃРёРјР°Р»СЊРЅРѕР№ Р°РјРїР»РёС‚СѓРґС‹ (Р±С‹Р»Рѕ 0.010)

# Р¤Р°РЅС‚РѕРјРЅС‹Рµ С„СЂР°Р·С‹, РєРѕС‚РѕСЂС‹Рµ С‡Р°СЃС‚Рѕ РіРµРЅРµСЂРёСЂСѓРµС‚ Whisper РёР· С€СѓРјР°
PHANTOM_PHRASES = [
    "СЂРµРґР°РєС‚РѕСЂ СЃСѓР±С‚РёС‚СЂРѕРІ",
    "РїРѕРґРїРёСЃС‹РІР°Р№С‚РµСЃСЊ РЅР° РЅР°С€ РєР°РЅР°Р»",
    "СѓСѓСѓСѓСѓСѓ",
    "СЃСѓР±С‚РёС‚СЂРѕРІ",
    "С‚РµР»РµС„РѕРЅРЅС‹Р№ Р·РІРѕРЅРѕРє",
    "Р·РІРѕРЅРѕРє",
    "СЃСѓР±С‚РёС‚СЂС‹ СЃРѕР·РґР°РІР°Р»",  # РљР»Р°СЃСЃРёС‡РµСЃРєР°СЏ РіР°Р»Р»СЋС†РёРЅР°С†РёСЏ Whisper РЅР° С‚РёС€РёРЅРµ
    "dimatorzok",  # Р’Р°СЂРёР°С†РёСЏ С‚СЂР°РЅСЃР»РёС‚РµСЂР°С†РёРё
    "dima torzok",
]

# РџР°С‚С‚РµСЂРЅС‹ РґР»СЏ РѕР±РЅР°СЂСѓР¶РµРЅРёСЏ С„Р°РЅС‚РѕРјРЅС‹С… Р·РІСѓРєРѕРІ (СЌРјРјРј, Р°Р°Р°, РїРѕРІС‚РѕСЂСЏСЋС‰РёРµСЃСЏ СЃР»РѕРІР°)
PHANTOM_PATTERNS = [
    r'^СЌРј+[РјРј]*$',           # СЌРј, СЌРјРј, СЌРјРјРјРј
    r'^РјРј+[РјРј]*$',           # РјРј, РјРјРј
    r'^Р°Р°+[Р°Р°]*$',           # Р°Р°, Р°Р°Р°
    r'^РєРёРІРё[.\s]*РєРёРІРё[.\s]*РєРёРІРё',  # РїРѕРІС‚РѕСЂСЏСЋС‰РёРµСЃСЏ РєРёРІРё
    r'^(РєРёРІРё[.\s]*){2,}$',   # С‚РѕР»СЊРєРѕ РєРёРІРё РїРѕРІС‚РѕСЂСЏРµС‚СЃСЏ 2+ СЂР°Р·Р°
]


# === РЎРўР РРњРРќР“РћР’РђРЇ РўР РђРќРЎРљР РРџР¦РРЇ (NEW) ===
class StreamingTranscriber:
    """
    РРЅРєСЂРµРјРµРЅС‚Р°Р»СЊРЅР°СЏ С‚СЂР°РЅСЃРєСЂРёРїС†РёСЏ Р°СѓРґРёРѕ.
    РќР°РєР°РїР»РёРІР°РµС‚ Р°СѓРґРёРѕ Рё РїРµСЂРёРѕРґРёС‡РµСЃРєРё Р·Р°РїСѓСЃРєР°РµС‚ Whisper РЅР° РЅР°РєРѕРїР»РµРЅРЅРѕРј Р±СѓС„РµСЂРµ.
    
    Thread-safe: audio_buffer Р·Р°С‰РёС‰С‘РЅ threading.Lock.
    """
    
    def __init__(self, model: WhisperModel, sample_rate: int = 16000, 
                 chunk_interval: float = 1.5, min_audio_for_stream: float = 1.0):
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_interval = chunk_interval
        self.min_audio_for_stream = min_audio_for_stream
        self._audio_buffer = []
        self._buffer_lock = threading.Lock()  # Р—Р°С‰РёС‚Р° Р±СѓС„РµСЂР° РѕС‚ race condition
        self.last_transcription = ""
        self.last_transcription_time = 0.0
        self.transcription_in_progress = False  # Р—Р°С‰РёС‚Р° РѕС‚ РїР°СЂР°Р»Р»РµР»СЊРЅС‹С… Р·Р°РїСѓСЃРєРѕРІ
        
    def add_audio(self, audio_chunk: np.ndarray):
        """Р”РѕР±Р°РІР»СЏРµС‚ С‡Р°РЅРє Р°СѓРґРёРѕ РІ Р±СѓС„РµСЂ (thread-safe)."""
        with self._buffer_lock:
            self._audio_buffer.append(audio_chunk.copy())
    
    def get_audio_for_transcription(self) -> Optional[np.ndarray]:
        """
        Р‘РµР·РѕРїР°СЃРЅРѕ РїРѕР»СѓС‡Р°РµС‚ РєРѕРїРёСЋ Р±СѓС„РµСЂР° РґР»СЏ С‚СЂР°РЅСЃРєСЂРёР±Р°С†РёРё.
        Returns: numpy array РёР»Рё None
        """
        with self._buffer_lock:
            if not self._audio_buffer:
                return None
            # РЎРѕР·РґР°С‘Рј РєРѕРїРёСЋ Р±СѓС„РµСЂР°
            audio = np.concatenate(self._audio_buffer.copy())
            return audio
        
    def should_transcribe(self) -> bool:
        """РџСЂРѕРІРµСЂСЏРµС‚, РїРѕСЂР° Р»Рё Р·Р°РїСѓСЃРєР°С‚СЊ С‚СЂР°РЅСЃРєСЂРёР±Р°С†РёСЋ (thread-safe)."""
        # РќРµ Р·Р°РїСѓСЃРєР°С‚СЊ РµСЃР»Рё СѓР¶Рµ РёРґС‘С‚ С‚СЂР°РЅСЃРєСЂРёР±Р°С†РёСЏ
        if self.transcription_in_progress:
            return False
        
        with self._buffer_lock:
            buffer_len = len(self._audio_buffer)
        
        total_duration = buffer_len * CHUNK_DURATION
        time_since_last = time.time() - self.last_transcription_time
        
        # РўСЂР°РЅСЃРєСЂРёР±РёСЂСѓРµРј РµСЃР»Рё:
        # 1. РќР°РєРѕРїРёР»РѕСЃСЊ РґРѕСЃС‚Р°С‚РѕС‡РЅРѕ Р°СѓРґРёРѕ (min_audio_for_stream)
        # 2. РџСЂРѕС€С‘Р» РёРЅС‚РµСЂРІР°Р» СЃ РїРѕСЃР»РµРґРЅРµР№ С‚СЂР°РЅСЃРєСЂРёР±Р°С†РёРё
        return (total_duration >= self.min_audio_for_stream and 
                time_since_last >= self.chunk_interval)
    
    def transcribe_partial(self, fix_callback=None) -> Optional[str]:
        """
        РўСЂР°РЅСЃРєСЂРёР±РёСЂСѓРµС‚ РЅР°РєРѕРїР»РµРЅРЅРѕРµ Р°СѓРґРёРѕ.
        Р’РѕР·РІСЂР°С‰Р°РµС‚ С‚РµРєСЃС‚ РёР»Рё None РµСЃР»Рё РЅРµС‚ СЂРµС‡Рё.
        
        Args:
            fix_callback: Р¤СѓРЅРєС†РёСЏ РґР»СЏ РёСЃРїСЂР°РІР»РµРЅРёСЏ С‚СЂР°РЅСЃРєСЂРёРїС†РёРё (_fix_transcription)
        """
        if self.transcription_in_progress:
            return None
        
        # РџРѕР»СѓС‡Р°РµРј Р°СѓРґРёРѕ Р±РµР·РѕРїР°СЃРЅРѕ (СЃРѕР·РґР°С‘Рј РєРѕРїРёСЋ)
        audio = self.get_audio_for_transcription()
        if audio is None:
            return None
        
        self.transcription_in_progress = True
        
        try:
            duration = len(audio) / self.sample_rate
            
            # РЎР»РёС€РєРѕРј РєРѕСЂРѕС‚РєРѕРµ Р°СѓРґРёРѕ вЂ” РїСЂРѕРїСѓСЃРєР°РµРј
            if duration < 0.4:
                self.transcription_in_progress = False
                return None
            
            # РћРїС‚РёРјРёР·РёСЂРѕРІР°РЅРЅС‹Рµ РїР°СЂР°РјРµС‚СЂС‹ РґР»СЏ speed > accuracy (partial transcribe)
            segments, info = self.model.transcribe(
                audio,
                language="ru",
                task="transcribe",
                beam_size=1,           # Р‘С‹СЃС‚СЂРµРµ С‡РµРј 5
                best_of=1,             # Р‘С‹СЃС‚СЂРµРµ С‡РµРј 5
                condition_on_previous_text=False,
                initial_prompt=WHISPER_INITIAL_PROMPT,
                no_speech_threshold=0.7,
                compression_ratio_threshold=2.4,  # РђРіСЂРµСЃСЃРёРІРЅРµРµ С„РёР»СЊС‚СЂСѓРµРј РјСѓСЃРѕСЂ
            )
            
            text_parts = []
            for segment in segments:
                # Р¤РёР»СЊС‚СЂСѓРµРј СЃРµРіРјРµРЅС‚С‹ СЃ РІС‹СЃРѕРєРѕР№ РІРµСЂРѕСЏС‚РЅРѕСЃС‚СЊСЋ "РЅРµС‚ СЂРµС‡Рё"
                no_speech = getattr(segment, 'no_speech_prob', 0.0)
                if no_speech < 0.7:
                    text_parts.append(segment.text)
            
            text = " ".join(text_parts).strip()
            
            # РџСЂРёРјРµРЅСЏРµРј Р°РІС‚РѕРёСЃРїСЂР°РІР»РµРЅРёРµ РµСЃР»Рё РґРѕСЃС‚СѓРїРЅРѕ
            if text and fix_callback:
                text = fix_callback(text)
            
            self.last_transcription = text
            self.last_transcription_time = time.time()
            
            return text if text else None
            
        except Exception as e:
            kiwi_log("STREAM", f"Transcription error: {e}", level="ERROR")
            return None
        finally:
            self.transcription_in_progress = False
    
    def clear(self):
        """РћС‡РёС‰Р°РµС‚ Р±СѓС„РµСЂ РїРѕСЃР»Рµ РѕР±СЂР°Р±РѕС‚РєРё РєРѕРјР°РЅРґС‹ (thread-safe)."""
        with self._buffer_lock:
            self._audio_buffer = []
        self.last_transcription = ""


@dataclass
class ListenerConfig:
    model_name: str = "small"
    device: str = "cuda"
    compute_type: str = "float16"
    wake_word: str = WAKE_WORD
    position_limit: int = POSITION_LIMIT
    sample_rate: int = SAMPLE_RATE
    # Realtime tuning defaults (overridden by config.yaml realtime section)
    chunk_duration: float = CHUNK_DURATION
    pre_buffer_duration: float = PRE_BUFFER_DURATION
    noise_sample_duration: float = NOISE_SAMPLE_DURATION
    noise_threshold_multiplier: float = NOISE_THRESHOLD_MULTIPLIER
    min_speech_duration: float = MIN_SPEECH_DURATION
    max_speech_duration: float = MAX_SPEECH_DURATION
    silence_duration_end: float = SILENCE_DURATION_END
    min_speech_volume: float = MIN_SPEECH_VOLUME


class WakeWordDetector:
    """РЈРјРЅС‹Р№ РґРµС‚РµРєС‚РѕСЂ wake word."""

    def __init__(self, wake_word: str = WAKE_WORD, position_limit: int = POSITION_LIMIT):
        normalized = (wake_word or "").strip().lower()
        # Защита от битой кодировки дефолта в исходнике.
        if not normalized or re.search(r"[а-яё]", normalized, re.IGNORECASE) is None:
            normalized = "киви"
        self.wake_word = normalized
        self.position_limit = position_limit
    
    def is_direct_address(self, text: str) -> Tuple[bool, Optional[str]]:
        text = text.strip().lower()
        if not text:
            return False, None

        word_matches = list(re.finditer(r"\w+", text, flags=re.UNICODE))
        wake_word_position = None
        wake_word_end = None
        for i, match in enumerate(word_matches[: self.position_limit]):
            clean_word = match.group(0)
            if clean_word == self.wake_word:
                wake_word_position = i
                wake_word_end = match.end()
                break
            similarity = difflib.SequenceMatcher(None, clean_word, self.wake_word).ratio()
            if similarity >= FUZZY_MATCH_THRESHOLD:
                # РџСЂРѕРІРµСЂРєР° РЅР° С‡С‘СЂРЅС‹Р№ СЃРїРёСЃРѕРє (С‡Р°СЃС‚С‹Рµ Р»РѕР¶РЅС‹Рµ СЃСЂР°Р±Р°С‚С‹РІР°РЅРёСЏ)
                if clean_word in FUZZY_BLACKLIST:
                    kiwi_log("WAKE", f"Fuzzy match ignored: '{clean_word}' in blacklist (sim={similarity:.2f})")
                    continue
                wake_word_position = i
                wake_word_end = match.end()
                kiwi_log("WAKE", f"Fuzzy match: '{clean_word}' ~ '{self.wake_word}' (sim={similarity:.2f})")
                break

        if wake_word_position is None or wake_word_end is None:
            return False, None

        # Простейшая защита от упоминаний вроде "спроси у киви".
        if wake_word_position > 0:
            prev = word_matches[wake_word_position - 1].group(0)
            if prev in {"у", "от", "для", "про", "о", "об", "спроси", "скажи"}:
                return False, None

        if wake_word_position > 2:
            return False, None

        command = self._extract_command(text, wake_word_end)
        return True, command

    def _extract_command(self, text: str, wake_word_end: int) -> Optional[str]:
        command = text[wake_word_end:].strip(" \t\n\r,.:;!?-")
        if not command:
            return None
        # Отбрасываем "команды" из одной пунктуации (например "."), чтобы
        # не триггерить early wake от эха/шумов.
        if not re.search(r"[0-9A-Za-zА-Яа-яЁё]", command):
            return None
        return command


class KiwiListener:
    """РЎР»СѓС€Р°С‚РµР»СЊ РЅР° Р±Р°Р·Рµ Faster Whisper СЃ СЂРµР¶РёРјРѕРј РґРёР°Р»РѕРіР°.
    
    РћРїС‚РёРјРёР·Р°С†РёРё РґР»СЏ realtime:
    - РђРґР°РїС‚РёРІРЅС‹Р№ РїРѕСЂРѕРі С€СѓРјР° (РєР°Р»РёР±СЂСѓРµС‚СЃСЏ РїСЂРё СЃС‚Р°СЂС‚Рµ)
    - VAD С‡РµСЂРµР· Whisper (Р±РѕР»РµРµ С‚РѕС‡РЅРѕРµ РѕРїСЂРµРґРµР»РµРЅРёРµ СЂРµС‡Рё)
    - Р‘С‹СЃС‚СЂР°СЏ РѕР±СЂР°Р±РѕС‚РєР° СЃ РѕРїС‚РёРјР°Р»СЊРЅС‹РјРё РїР°СЂР°РјРµС‚СЂР°РјРё Whisper
    - Voice Priority Queue (OWNER РїСЂРµСЂС‹РІР°РµС‚ GUEST)
    """
    
    def __init__(
        self,
        config: Optional[ListenerConfig] = None,
        on_wake_word: Optional[Callable[[str], None]] = None,
        on_speech: Optional[Callable[[str], None]] = None,
        on_dialog_timeout: Optional[Callable[[], None]] = None,
        dialog_timeout: float = 5.0
    ):
        self.config = config or ListenerConfig()
        self.on_wake_word = on_wake_word
        self.on_speech = on_speech
        self.on_dialog_timeout = on_dialog_timeout
        self.dialog_timeout = dialog_timeout
        
        self.model = None
        self.detector = WakeWordDetector(
            wake_word=self.config.wake_word,
            position_limit=self.config.position_limit
        )
        
        self.audio_queue: queue.Queue = queue.Queue()
        self.is_running = False
        self._recording_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._dialog_timeout_thread: Optional[threading.Thread] = None
        
        # Р РµР¶РёРј РґРёР°Р»РѕРіР°
        self.dialog_mode = False
        self.dialog_until = 0.0
        self._idle_played = False  # Р¤Р»Р°Рі С‡С‚Рѕ idle Р·РІСѓРє СѓР¶Рµ РїСЂРѕРёРіСЂР°РЅ
        
        # Р¤Р»Р°Рі: СЂРµС‡СЊ Р±С‹Р»Р° РЅР°С‡Р°С‚Р° РІ СЂРµР¶РёРјРµ РґРёР°Р»РѕРіР° (РґР»СЏ РѕР±СЂР°Р±РѕС‚РєРё РїРѕСЃР»Рµ С‚Р°Р№РјР°СѓС‚Р°)
        self._speech_started_in_dialog = False
        
        # === REALTIME: РђРґР°РїС‚РёРІРЅС‹Р№ РїРѕСЂРѕРі С€СѓРјР° ===
        self._noise_floor = 0.0  # РЈСЂРѕРІРµРЅСЊ С€СѓРјР° РѕРєСЂСѓР¶РµРЅРёСЏ
        self._silence_threshold = 0.015  # Р‘СѓРґРµС‚ РєР°Р»РёР±СЂРѕРІР°С‚СЊСЃСЏ
        
        # === MUTE РґР»СЏ РїСЂРµРґРѕС‚РІСЂР°С‰РµРЅРёСЏ self-listening ===
        self._is_muted = False  # РњРёРєСЂРѕС„РѕРЅ Р·Р°РјСЊСЋС‡РµРЅ РІРѕ РІСЂРµРјСЏ TTS
        
        # === LLM РґР»СЏ РїРѕСЃС‚-РѕР±СЂР°Р±РѕС‚РєРё С‚СЂР°РЅСЃРєСЂРёРїС†РёРё ===
        self._llm_fix_callback: Optional[Callable[[str], str]] = None
        
        # === РђР”РђРџРўРР’РќРђРЇ РџРђРЈР—Рђ Р”Р›РЇ Р”Р›РРќРќР«РҐ Р¤Р РђР— ===
        # Р‘СѓРґСѓС‚ Р·Р°РіСЂСѓР¶РµРЅС‹ РёР· config.yaml РІ _load_streaming_config()
        self._silence_duration_long_speech = 2.5    # РџР°СѓР·Р° РґР»СЏ РґР»РёРЅРЅС‹С… С„СЂР°Р· (>3s)
        self._silence_duration_monologue = 3.5      # РџР°СѓР·Р° РґР»СЏ РјРѕРЅРѕР»РѕРіРѕРІ (>8s)
        self._long_speech_threshold = 3.0           # РџРѕСЂРѕРі "РґР»РёРЅРЅРѕР№ С„СЂР°Р·С‹"
        self._monologue_threshold = 8.0             # РџРѕСЂРѕРі "РјРѕРЅРѕР»РѕРіР°"
        
        # === VAD-РЈРЎРР›Р•РќРќРћР• РћРџР Р•Р”Р•Р›Р•РќРР• РљРћРќР¦Рђ Р Р•Р§Р ===
        self._vad_end_speech_check = True           # РСЃРїРѕР»СЊР·РѕРІР°С‚СЊ VAD РґР»СЏ РїРѕРґС‚РІРµСЂР¶РґРµРЅРёСЏ РєРѕРЅС†Р°
        self._vad_end_speech_frames = 5             # РЎРєРѕР»СЊРєРѕ VAD-С‡Р°РЅРєРѕРІ Р±РµР· СЂРµС‡Рё РЅСѓР¶РЅРѕ
        self._vad_continuation_threshold = 3        # РЎРєРѕР»СЊРєРѕ С‡Р°РЅРєРѕРІ СЃ СЂРµС‡СЊСЋ РёР· РїРѕСЃР»РµРґРЅРёС… N РґР»СЏ РїСЂРѕРґРѕР»Р¶РµРЅРёСЏ
        self._vad_continuation_bonus_chunks = 5     # Р‘РѕРЅСѓСЃ Рє РїР°СѓР·Рµ РµСЃР»Рё VAD РІРёРґРёС‚ РїСЂРѕРґРѕР»Р¶РµРЅРёРµ СЂРµС‡Рё
        
        # === SPEAKER IDENTIFICATION (Phase 1: echo cancellation) ===
        self.speaker_id: Optional[SpeakerIdentifier] = None
        if SPEAKER_ID_AVAILABLE:
            try:
                self.speaker_id = SpeakerIdentifier()
                kiwi_log("SPEAKER", "Speaker identification initialized")
            except Exception as e:
                kiwi_log("SPEAKER", f"Failed to initialize speaker ID: {e}", level="ERROR")
        
        # === SPEAKER MANAGER (Priority + Access Control) ===
        self.speaker_manager: Optional[SpeakerManager] = None
        if SPEAKER_MANAGER_AVAILABLE:
            try:
                self.speaker_manager = SpeakerManager(base_identifier=self.speaker_id)
                kiwi_log("SPEAKER_MANAGER", "Initialized")
            except Exception as e:
                kiwi_log("SPEAKER_MANAGER", f"Failed to initialize: {e}", level="ERROR")
        
        # === VOICE PRIORITY QUEUE ===
        self._voice_queue: queue.Queue = queue.Queue()  # (audio, speaker_id, priority)
        self._current_speaker_id: Optional[str] = None
        self._current_priority: int = 999
        self._owner_override_event: threading.Event = threading.Event()
        
        # === VOICE SECURITY ===
        self._voice_security: Optional[VoiceSecurity] = None
        if VOICE_SECURITY_AVAILABLE:
            try:
                self._voice_security = VoiceSecurity()
                kiwi_log("VOICE_SECURITY", "Initialized")
            except Exception as e:
                kiwi_log("VOICE_SECURITY", f"Failed to initialize: {e}", level="ERROR")

        # === SPEAKER CONTEXT SNAPSHOT (for service-side policy checks) ===
        self._speaker_meta_lock = threading.Lock()
        self._last_speaker_meta: Dict[str, Any] = {
            "speaker_id": "unknown",
            "speaker_name": "РќРµР·РЅР°РєРѕРјРµС†",
            "priority": int(VoicePriority.GUEST) if SPEAKER_MANAGER_AVAILABLE else 2,
            "confidence": 0.0,
            "music_probability": 0.0,
            "text": "",
            "timestamp": 0.0,
        }
        self._last_utterance_audio: Optional[np.ndarray] = None

        # === BACKGROUND MUSIC FILTER ===
        self._music_filter_enabled = True
        self._music_reject_threshold = 0.78
        
        # === OWNER HANDS-FREE POLICY ===
        # Owner can talk without wake word in quiet 1:1 mode.
        # This is automatically disabled when guests are active or music is likely present.
        self._owner_handsfree_enabled = True
        self._owner_handsfree_guest_cooldown = 180.0
        self._last_non_owner_activity_at = 0.0
        
        # === SILERO VAD: Р‘С‹СЃС‚СЂР°СЏ Р»РѕРєР°Р»СЊРЅР°СЏ С„РёР»СЊС‚СЂР°С†РёСЏ СЂРµС‡СЊ/С€СѓРј ===
        self._vad_model = None
        self._vad_enabled = False
        self._vad_loaded = False
        if TORCH_AVAILABLE:
            kiwi_log("VAD", "Silero VAD ready (will load at start)")
        else:
            kiwi_log("VAD", "torch not installed — using energy-only barge-in")
        
        # === BARGE-IN: РЎРѕСЃС‚РѕСЏРЅРёРµ РґР»СЏ СѓРјРЅРѕРіРѕ РѕРїСЂРµРґРµР»РµРЅРёСЏ РїСЂРµСЂС‹РІР°РЅРёСЏ ===
        self._barge_in_counter = 0          # РЎС‡С‘С‚С‡РёРє РїРѕРґСЂСЏРґ РёРґСѓС‰РёС… С‡Р°РЅРєРѕРІ СЃ СЂРµС‡СЊСЋ
        self._barge_in_chunks_required = 5  # РќСѓР¶РЅРѕ 5 С‡Р°РЅРєРѕРІ РїРѕРґСЂСЏРґ (~1.5s) РґР»СЏ barge-in
        self._barge_in_volume_multiplier = 5.0  # РџРѕСЂРѕРі РіСЂРѕРјРєРѕСЃС‚Рё Г—5.0 РІРѕ РІСЂРµРјСЏ TTS
        self._barge_in_min_volume = 0.045   # РђР±СЃРѕР»СЋС‚РЅС‹Р№ РјРёРЅРёРјСѓРј РіСЂРѕРјРєРѕСЃС‚Рё РґР»СЏ barge-in
        self._barge_in_grace_period = 0.35  # 350ms РїРѕСЃР»Рµ РЅР°С‡Р°Р»Р° TTS вЂ” РёРіРЅРѕСЂРёСЂСѓРµРј barge-in
        self._tts_start_time = 0.0          # РњРµС‚РєР° РЅР°С‡Р°Р»Р° TTS (СѓСЃС‚Р°РЅР°РІР»РёРІР°РµС‚СЃСЏ РёР· service)
        self._post_tts_dead_zone = 2.5      # 2.5СЃ РїРѕСЃР»Рµ Р·Р°РІРµСЂС€РµРЅРёСЏ TTS вЂ” РЅРµ Р·Р°РїРёСЃС‹РІР°РµРј (СЌС…Рѕ Р·Р°С‚РёС…Р°РµС‚)
        
        # === Р”Р•Р‘РђРЈРќРЎРРќР“ Рё РљРћРќРўР РћР›Р¬ РћР§Р•Р Р•Р”Р ===
        self._last_submit_time = 0.0        # Р’СЂРµРјСЏ РїРѕСЃР»РµРґРЅРµРіРѕ submit
        self._submit_debounce = 0.5         # РњРёРЅРёРјСѓРј 500ms РјРµР¶РґСѓ submit
        self._last_audio_status_log = 0.0   # РўСЂРѕС‚С‚Р»РёРЅРі Р»РѕРіРѕРІ audio callback
        self._input_overflow_count = 0      # РЎС‡С‘С‚С‡РёРє input overflow РґР»СЏ РґРёР°РіРЅРѕСЃС‚РёРєРё
        
        # === РњР•РўРљР Р”Р›РЇ РљРћР РћРўРљРРҐ Р—Р’РЈРљРћР’ (idle, beep) ===
        self._sound_end_time = 0.0          # Р’СЂРµРјСЏ Р·Р°РІРµСЂС€РµРЅРёСЏ РєРѕСЂРѕС‚РєРѕРіРѕ Р·РІСѓРєР°
        self._post_sound_dead_zone = 0.2    # 200ms вЂ” РґРѕСЃС‚Р°С‚РѕС‡РЅРѕ РґР»СЏ Р·Р°С‚СѓС…Р°РЅРёСЏ idle/beep
        
        # === РЎРўР РРњРРќР“РћР’РђРЇ РўР РђРќРЎРљР РРџР¦РРЇ (NEW) ===
        self._streaming_enabled = False
        self._streaming_transcriber: Optional[StreamingTranscriber] = None
        self._early_wake_detected = False
        self._early_wake_lock = threading.Lock()  # Р—Р°С‰РёС‚Р° РґР»СЏ early wake С„Р»Р°РіРѕРІ
        self._early_command = ""
        self._early_detected_at = 0.0
        self._streaming_early_timeout = 3.0  # 3 СЃРµРєСѓРЅРґС‹ Р°РєС‚СѓР°Р»СЊРЅРѕСЃС‚Рё early detection
        self._streaming_chunk_interval = 1.5
        self._streaming_min_audio = 1.0
        self._streaming_thread: Optional[threading.Thread] = None
        self._streaming_stop_event = threading.Event()

        # Instance copies from ListenerConfig (overridden by config.yaml)
        self._silence_duration_end = self.config.silence_duration_end
        self._min_speech_volume = self.config.min_speech_volume
        self._max_speech_duration = self.config.max_speech_duration

        # Р—Р°РіСЂСѓР¶Р°РµРј РєРѕРЅС„РёРі СЃС‚СЂРёРјРёРЅРіР°
        self._load_streaming_config()
    
    def _load_streaming_config(self):
        """Р—Р°РіСЂСѓР¶Р°РµС‚ РЅР°СЃС‚СЂРѕР№РєРё СЃС‚СЂРёРјРёРЅРіР° Рё VAD РёР· config.yaml"""
        import yaml
        # Formerly: global SILENCE_DURATION_END, MIN_SPEECH_VOLUME, MAX_SPEECH_DURATION
        try:
            from kiwi import PROJECT_ROOT
            config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                # Р—Р°РіСЂСѓР¶Р°РµРј realtime РЅР°СЃС‚СЂРѕР№РєРё
                if config and 'realtime' in config:
                    realtime_config = config['realtime']
                    
                    # Р—Р°РіСЂСѓР¶Р°РµРј silence_duration_end РёР· РєРѕРЅС„РёРіР°
                    if 'silence_duration_end' in realtime_config:
                        self._silence_duration_end = realtime_config['silence_duration_end']
                        kiwi_log("CONFIG", f"Loaded silence_duration_end={self._silence_duration_end}s from config")
                    
                    # Р—Р°РіСЂСѓР¶Р°РµРј max_speech_duration РёР· РєРѕРЅС„РёРіР°
                    if 'max_speech_duration' in realtime_config:
                        self._max_speech_duration = realtime_config['max_speech_duration']
                        kiwi_log("CONFIG", f"Loaded max_speech_duration={self._max_speech_duration}s from config")
                    
                    # Р—Р°РіСЂСѓР¶Р°РµРј min_speech_volume РёР· РєРѕРЅС„РёРіР°
                    if 'min_speech_volume' in realtime_config:
                        self._min_speech_volume = realtime_config['min_speech_volume']
                        kiwi_log("CONFIG", f"Loaded min_speech_volume={self._min_speech_volume} from config")
                    
                    # Р—Р°РіСЂСѓР¶Р°РµРј post_tts_dead_zone РёР· РєРѕРЅС„РёРіР°
                    if 'post_tts_dead_zone' in realtime_config:
                        self._post_tts_dead_zone = realtime_config['post_tts_dead_zone']
                        kiwi_log("CONFIG", f"Loaded post_tts_dead_zone={self._post_tts_dead_zone}s from config")

                    # РџР°СЂР°РјРµС‚СЂС‹ barge-in (РїСЂРµСЂС‹РІР°РЅРёРµ TTS РіРѕР»РѕСЃРѕРј)
                    if 'barge_in_chunks_required' in realtime_config:
                        self._barge_in_chunks_required = max(1, int(realtime_config['barge_in_chunks_required']))
                        kiwi_log("CONFIG", f"Loaded barge_in_chunks_required={self._barge_in_chunks_required}")

                    if 'barge_in_volume_multiplier' in realtime_config:
                        self._barge_in_volume_multiplier = max(1.0, float(realtime_config['barge_in_volume_multiplier']))
                        kiwi_log("CONFIG", f"Loaded barge_in_volume_multiplier={self._barge_in_volume_multiplier}")

                    if 'barge_in_min_volume' in realtime_config:
                        self._barge_in_min_volume = max(0.001, float(realtime_config['barge_in_min_volume']))
                        kiwi_log("CONFIG", f"Loaded barge_in_min_volume={self._barge_in_min_volume}")

                    if 'barge_in_grace_period' in realtime_config:
                        self._barge_in_grace_period = max(0.0, float(realtime_config['barge_in_grace_period']))
                        kiwi_log("CONFIG", f"Loaded barge_in_grace_period={self._barge_in_grace_period}s")
                    
                    # === РђР”РђРџРўРР’РќРђРЇ РџРђРЈР—Рђ ===
                    if 'silence_duration_long_speech' in realtime_config:
                        self._silence_duration_long_speech = realtime_config['silence_duration_long_speech']
                        kiwi_log("CONFIG", f"Loaded silence_duration_long_speech={self._silence_duration_long_speech}s from config")
                    
                    if 'silence_duration_monologue' in realtime_config:
                        self._silence_duration_monologue = realtime_config['silence_duration_monologue']
                        kiwi_log("CONFIG", f"Loaded silence_duration_monologue={self._silence_duration_monologue}s from config")
                    
                    if 'long_speech_threshold' in realtime_config:
                        self._long_speech_threshold = realtime_config['long_speech_threshold']
                        kiwi_log("CONFIG", f"Loaded long_speech_threshold={self._long_speech_threshold}s from config")
                    
                    if 'monologue_threshold' in realtime_config:
                        self._monologue_threshold = realtime_config['monologue_threshold']
                        kiwi_log("CONFIG", f"Loaded monologue_threshold={self._monologue_threshold}s from config")
                    
                    # === VAD РџРђР РђРњР•РўР Р« ===
                    if 'vad_end_speech_check' in realtime_config:
                        self._vad_end_speech_check = realtime_config['vad_end_speech_check']
                        kiwi_log("CONFIG", f"Loaded vad_end_speech_check={self._vad_end_speech_check} from config")
                    
                    if 'vad_end_speech_frames' in realtime_config:
                        self._vad_end_speech_frames = realtime_config['vad_end_speech_frames']
                        kiwi_log("CONFIG", f"Loaded vad_end_speech_frames={self._vad_end_speech_frames} from config")
                    
                    if 'vad_continuation_threshold' in realtime_config:
                        self._vad_continuation_threshold = realtime_config['vad_continuation_threshold']
                        kiwi_log("CONFIG", f"Loaded vad_continuation_threshold={self._vad_continuation_threshold} from config")
                    
                    if 'vad_continuation_bonus_chunks' in realtime_config:
                        self._vad_continuation_bonus_chunks = realtime_config['vad_continuation_bonus_chunks']
                        kiwi_log("CONFIG", f"Loaded vad_continuation_bonus_chunks={self._vad_continuation_bonus_chunks} from config")
                    
                    # Р—Р°РіСЂСѓР¶Р°РµРј streaming РЅР°СЃС‚СЂРѕР№РєРё
                    if 'streaming' in realtime_config:
                        streaming_config = realtime_config['streaming']
                        self._streaming_enabled = streaming_config.get('enabled', True)
                        self._streaming_chunk_interval = streaming_config.get('chunk_interval', 1.5)
                        self._streaming_min_audio = streaming_config.get('min_audio_for_stream', 1.0)
                        kiwi_log("STREAMING", f"Config loaded: enabled={self._streaming_enabled}, interval={self._streaming_chunk_interval}s, min_audio={self._streaming_min_audio}s")

                    # РќР°СЃС‚СЂРѕР№РєРё С„РёР»СЊС‚СЂР° С„РѕРЅРѕРІРѕР№ РјСѓР·С‹РєРё
                    if 'music_filter' in realtime_config:
                        music_config = realtime_config['music_filter']
                        self._music_filter_enabled = music_config.get('enabled', self._music_filter_enabled)
                        self._music_reject_threshold = music_config.get('reject_threshold', self._music_reject_threshold)
                        kiwi_log("MUSIC", f"Config loaded: enabled={self._music_filter_enabled}, reject_threshold={self._music_reject_threshold:.2f}")
                    
                    if 'owner_handsfree' in realtime_config:
                        handsfree_config = realtime_config['owner_handsfree']
                        self._owner_handsfree_enabled = bool(
                            handsfree_config.get('enabled', self._owner_handsfree_enabled)
                        )
                        self._owner_handsfree_guest_cooldown = max(
                            0.0,
                            float(handsfree_config.get('guest_cooldown_sec', self._owner_handsfree_guest_cooldown))
                        )
                        kiwi_log("HANDSFREE", f"Config loaded: enabled={self._owner_handsfree_enabled}, guest_cooldown={self._owner_handsfree_guest_cooldown:.0f}s")
                else:
                    kiwi_log("STREAMING", "No realtime config found, using defaults")
            else:
                kiwi_log("STREAMING", f"Config file not found: {config_path}", level="WARNING")
        except Exception as e:
            kiwi_log("STREAMING", f"Error loading config: {e}, using defaults", level="ERROR")
            self._streaming_enabled = True
    
    def _get_silence_duration(self, speech_duration: float) -> float:
        """РђРґР°РїС‚РёРІРЅР°СЏ РїР°СѓР·Р°: С‡РµРј РґРѕР»СЊС€Рµ СЂРµС‡СЊ, С‚РµРј Р±РѕР»СЊС€Рµ РґРѕРїСѓСЃС‚РёРјР°СЏ РїР°СѓР·Р°.
        
        Args:
            speech_duration: Р”Р»РёС‚РµР»СЊРЅРѕСЃС‚СЊ С‚РµРєСѓС‰РµР№ СЂРµС‡Рё РІ СЃРµРєСѓРЅРґР°С…
            
        Returns:
            Р”Р»РёС‚РµР»СЊРЅРѕСЃС‚СЊ РїР°СѓР·С‹ (РІ СЃРµРєСѓРЅРґР°С…) РґР»СЏ Р·Р°РІРµСЂС€РµРЅРёСЏ СЂРµС‡Рё
        """
        if speech_duration >= self._monologue_threshold:
            return self._silence_duration_monologue  # 3.5s
        elif speech_duration >= self._long_speech_threshold:
            return self._silence_duration_long_speech  # 2.5s
        else:
            return self._silence_duration_end  # 1.5s (Р±Р°Р·РѕРІР°СЏ)

    def _get_effective_min_speech_volume(self) -> float:
        """РђРґР°РїС‚РёРІРЅС‹Р№ РјРёРЅРёРјСѓРј РіСЂРѕРјРєРѕСЃС‚Рё РґР»СЏ СЃС‚Р°СЂС‚Р° Р·Р°РїРёСЃРё СЂРµС‡Рё.
        
        Р–С‘СЃС‚РєРёР№ РїРѕСЂРѕРі 0.015 РјРѕР¶РµС‚ СЃРґРµР»Р°С‚СЊ СЃРёСЃС‚РµРјСѓ "РіР»СѓС…РѕР№" РЅР° С‚РёС…РёС… РјРёРєСЂРѕС„РѕРЅР°С….
        Р”РµР»Р°РµРј РїРѕСЂРѕРі Р·Р°РІРёСЃРёРјС‹Рј РѕС‚ РєР°Р»РёР±СЂРѕРІР°РЅРЅРѕРіРѕ noise floor, РЅРѕ РЅРµ РЅРёР¶Рµ 0.006.
        """
        noise_based_floor = self._silence_threshold * 2.5
        return min(self._min_speech_volume, max(0.006, noise_based_floor))

    def _estimate_music_probability(self, audio: np.ndarray) -> float:
        """
        Р­РІСЂРёСЃС‚РёРєР° РґР»СЏ РґРµС‚РµРєС‚Р° С„РѕРЅРѕРІРѕР№ РјСѓР·С‹РєРё.

        Р’РѕР·РІСЂР°С‰Р°РµС‚ Р·РЅР°С‡РµРЅРёРµ 0..1 (С‡РµРј РІС‹С€Рµ, С‚РµРј РІРµСЂРѕСЏС‚РЅРµРµ РјСѓР·С‹РєР°).
        """
        if not self._music_filter_enabled:
            return 0.0

        if audio is None or len(audio) < int(0.8 * self.config.sample_rate):
            return 0.0

        try:
            x = audio.astype(np.float32, copy=False)
            max_abs = float(np.max(np.abs(x))) if len(x) else 0.0
            if max_abs > 0:
                x = x / max_abs

            frame = 1024
            hop = 512
            if len(x) < frame:
                return 0.0

            energies = []
            zcrs = []
            flatness = []
            flux = []
            prev_mag = None

            for i in range(0, len(x) - frame, hop):
                w = x[i:i + frame]
                if len(w) < frame:
                    continue

                # Р­РЅРµСЂРіРёСЏ Рё ZCR
                energies.append(float(np.sqrt(np.mean(w ** 2))))
                signs = np.sign(w)
                zcrs.append(float(np.mean(np.abs(np.diff(signs)) > 0)))

                # РЎРїРµРєС‚СЂР°Р»СЊРЅС‹Рµ РїСЂРёР·РЅР°РєРё
                mag = np.abs(np.fft.rfft(w)) + 1e-10
                gm = float(np.exp(np.mean(np.log(mag))))
                am = float(np.mean(mag))
                flatness.append(gm / (am + 1e-10))

                if prev_mag is not None:
                    num = np.linalg.norm(mag - prev_mag)
                    den = np.linalg.norm(prev_mag) + 1e-10
                    flux.append(float(num / den))
                prev_mag = mag

            if len(energies) < 5:
                return 0.0

            energy_mean = float(np.mean(energies)) + 1e-8
            energy_cv = float(np.std(energies) / energy_mean)
            zcr_mean = float(np.mean(zcrs)) if zcrs else 0.0
            flatness_mean = float(np.mean(flatness)) if flatness else 1.0
            flux_mean = float(np.mean(flux)) if flux else 1.0

            stable_energy = max(0.0, min(1.0, (0.35 - energy_cv) / 0.35))
            tonal = max(0.0, min(1.0, (0.25 - flatness_mean) / 0.25))
            zcr_centered = max(0.0, 1.0 - min(abs(zcr_mean - 0.08) / 0.08, 1.0))
            low_flux = max(0.0, min(1.0, (0.12 - flux_mean) / 0.12))

            score = (
                0.35 * stable_energy +
                0.35 * tonal +
                0.20 * zcr_centered +
                0.10 * low_flux
            )
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            kiwi_log("MUSIC", f"Detection error: {e}", level="ERROR")
            return 0.0

    def _update_last_speaker_meta(self, meta: Dict[str, Any], audio: Optional[np.ndarray] = None):
        """РћР±РЅРѕРІР»СЏРµС‚ РїРѕСЃР»РµРґРЅРёР№ СЃРЅРёРјРѕРє РіРѕРІРѕСЂСЏС‰РµРіРѕ РґР»СЏ service policy."""
        safe_meta = dict(meta or {})
        safe_meta.setdefault("speaker_id", "unknown")
        safe_meta.setdefault("speaker_name", "РќРµР·РЅР°РєРѕРјРµС†")
        safe_meta.setdefault("priority", int(VoicePriority.GUEST) if SPEAKER_MANAGER_AVAILABLE else 2)
        safe_meta.setdefault("confidence", 0.0)
        safe_meta.setdefault("music_probability", 0.0)
        safe_meta.setdefault("text", "")
        safe_meta["timestamp"] = time.time()

        with self._speaker_meta_lock:
            self._last_speaker_meta = safe_meta
            self._last_utterance_audio = audio.copy() if audio is not None else None

    def get_last_speaker_meta(self) -> Dict[str, Any]:
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ РјРµС‚Р°РґР°РЅРЅС‹Рµ РїРѕСЃР»РµРґРЅРµР№ СѓСЃРїРµС€РЅРѕ СЂР°СЃРїРѕР·РЅР°РЅРЅРѕР№ СЂРµРїР»РёРєРё."""
        with self._speaker_meta_lock:
            return dict(self._last_speaker_meta)

    def get_last_utterance_audio(self) -> Optional[np.ndarray]:
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ Р°СѓРґРёРѕ РїРѕСЃР»РµРґРЅРµР№ СЂРµРїР»РёРєРё (РµСЃР»Рё РµСЃС‚СЊ)."""
        with self._speaker_meta_lock:
            return None if self._last_utterance_audio is None else self._last_utterance_audio.copy()

    def register_owner_voice(self, name: str = "Р Р°РјРёР»СЊ") -> bool:
        """Р РµРіРёСЃС‚СЂРёСЂСѓРµС‚ РіРѕР»РѕСЃ РІР»Р°РґРµР»СЊС†Р° РїРѕ РїРѕСЃР»РµРґРЅРµР№ СЂРµРїР»РёРєРµ."""
        if self.speaker_manager is None:
            return False
        audio = self.get_last_utterance_audio()
        if audio is None or len(audio) < int(0.6 * self.config.sample_rate):
            return False
        return self.speaker_manager.register_owner(audio, self.config.sample_rate, name=name)

    def remember_last_voice_as(self, name: str) -> Tuple[bool, str]:
        """РЎРѕС…СЂР°РЅСЏРµС‚ РїРѕСЃР»РµРґРЅСЋСЋ СЂРµРїР»РёРєСѓ РєР°Рє РїСЂРѕС„РёР»СЊ Р·РЅР°РєРѕРјРѕРіРѕ."""
        if self.speaker_manager is None:
            return False, "Speaker manager not available"
        audio = self.get_last_utterance_audio()
        if audio is None or len(audio) < int(0.6 * self.config.sample_rate):
            return False, "No recent voice sample"
        return self.speaker_manager.add_friend(audio, self.config.sample_rate, name=name)

    def describe_last_speaker(self) -> str:
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ human-readable РѕРїРёСЃР°РЅРёРµ РїРѕСЃР»РµРґРЅРµРіРѕ РіРѕРІРѕСЂСЏС‰РµРіРѕ."""
        if self.speaker_manager is None:
            return "Р Р°СЃРїРѕР·РЅР°РІР°РЅРёРµ РіРѕРІРѕСЂСЏС‰РёС… РЅРµРґРѕСЃС‚СѓРїРЅРѕ."
        audio = self.get_last_utterance_audio()
        if audio is None:
            return "РќРµС‚ СЃРІРµР¶РµР№ СЂРµРїР»РёРєРё РґР»СЏ РёРґРµРЅС‚РёС„РёРєР°С†РёРё."
        return self.speaker_manager.who_am_i(audio, self.config.sample_rate)

    def describe_known_voices(self) -> str:
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ СЃРїРёСЃРѕРє РёР·РІРµСЃС‚РЅС‹С… РіРѕР»РѕСЃРѕРІ."""
        if self.speaker_manager is None:
            return "РЎРїРёСЃРѕРє РіРѕР»РѕСЃРѕРІ РЅРµРґРѕСЃС‚СѓРїРµРЅ."
        profiles = self.speaker_manager.get_profile_info()
        if not profiles:
            return "РџРѕРєР° РЅРµС‚ СЃРѕС…СЂР°РЅС‘РЅРЅС‹С… РіРѕР»РѕСЃРѕРІ."
        lines = ["РР·РІРµСЃС‚РЅС‹Рµ РіРѕР»РѕСЃР°:"]
        for _, info in profiles.items():
            status = " (Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ)" if info.get("is_blocked") else ""
            lines.append(f"- {info.get('name', 'Р‘РµР· РёРјРµРЅРё')}{status}")
        return "\n".join(lines)
    
    def _check_vad_continuation(self, audio_buffer: list) -> bool:
        """РџСЂРѕРІРµСЂСЏРµС‚ С‡РµСЂРµР· VAD, РїСЂРѕРґРѕР»Р¶Р°РµС‚СЃСЏ Р»Рё СЂРµС‡СЊ (РґР°Р¶Рµ РµСЃР»Рё РіСЂРѕРјРєРѕСЃС‚СЊ СѓРїР°Р»Р°).
        
        Args:
            audio_buffer: РЎРїРёСЃРѕРє Р°СѓРґРёРѕ-С‡Р°РЅРєРѕРІ
            
        Returns:
            True РµСЃР»Рё VAD РІРёРґРёС‚ РїСЂРѕРґРѕР»Р¶РµРЅРёРµ СЂРµС‡Рё (РЅРµ Р·Р°РІРµСЂС€Р°С‚СЊ Р·Р°РїРёСЃСЊ)
        """
        if not self._vad_enabled or not self._vad_end_speech_check:
            return False  # РќРµ РїСЂРѕРґРѕР»Р¶Р°РµРј (Р·Р°РІРµСЂС€Р°РµРј Р·Р°РїРёСЃСЊ)
        
        if len(audio_buffer) < self._vad_end_speech_frames:
            return False  # РќРµРґРѕСЃС‚Р°С‚РѕС‡РЅРѕ С‡Р°РЅРєРѕРІ РґР»СЏ РїСЂРѕРІРµСЂРєРё
        
        # РџСЂРѕРІРµСЂСЏРµРј РїРѕСЃР»РµРґРЅРёРµ N С‡Р°РЅРєРѕРІ С‡РµСЂРµР· VAD
        recent_chunks = audio_buffer[-self._vad_end_speech_frames:]
        vad_speech_count = 0
        
        for chunk in recent_chunks:
            if self._check_vad(chunk):
                vad_speech_count += 1
        
        # Р•СЃР»Рё Р±РѕР»СЊС€Рµ threshold С‡Р°РЅРєРѕРІ СЃРѕРґРµСЂР¶Р°С‚ СЂРµС‡СЊ вЂ” РїСЂРѕРґРѕР»Р¶Р°РµРј Р·Р°РїРёСЃСЊ
        return vad_speech_count >= self._vad_continuation_threshold
    
    def create_self_profile(self, tts_audio: np.ndarray, sample_rate: int = 24000) -> bool:
        """
        РЎРѕР·РґР°С‘С‚ РїСЂРѕС„РёР»СЊ СЃРѕР±СЃС‚РІРµРЅРЅРѕРіРѕ РіРѕР»РѕСЃР° (TTS) РґР»СЏ С„РёР»СЊС‚СЂР°С†РёРё СЌС…Р°.
        
        Args:
            tts_audio: Р°СѓРґРёРѕ СЃРіРµРЅРµСЂРёСЂРѕРІР°РЅРЅРѕРµ TTS
            sample_rate: С‡Р°СЃС‚РѕС‚Р° РґРёСЃРєСЂРµС‚РёР·Р°С†РёРё
            
        Returns:
            True РµСЃР»Рё СѓСЃРїРµС€РЅРѕ
        """
        if self.speaker_id is None:
            kiwi_log("SPEAKER", "Speaker ID not available", level="WARNING")
            return False

        try:
            success = self.speaker_id.create_self_profile(tts_audio, sample_rate)
            if success:
                kiwi_log("SPEAKER", "Self-profile created successfully")
            return success
        except Exception as e:
            kiwi_log("SPEAKER", f"Failed to create self-profile: {e}", level="ERROR")
            return False
    
    def calibrate_speaker_id(self) -> bool:
        """
        Р—Р°РїСѓСЃРєР°РµС‚ РєР°Р»РёР±СЂРѕРІРєСѓ speaker identification.
        Р“РµРЅРµСЂРёСЂСѓРµС‚ С‚РµСЃС‚РѕРІСѓСЋ С„СЂР°Р·Сѓ Рё СЃРѕР·РґР°С‘С‚ РїСЂРѕС„РёР»СЊ self.
        
        Returns:
            True РµСЃР»Рё СѓСЃРїРµС€РЅРѕ
        """
        if self.speaker_id is None:
            kiwi_log("SPEAKER", "Speaker ID not available for calibration", level="WARNING")
            return False

        kiwi_log("SPEAKER", "Starting TTS calibration...")
        return self.speaker_id.calibrate_self_from_tts()
    
    def set_llm_fix_callback(self, callback: Callable[[str], str]):
        """
        РЈСЃС‚Р°РЅР°РІР»РёРІР°РµС‚ callback РґР»СЏ LLM РёСЃРїСЂР°РІР»РµРЅРёСЏ С‚СЂР°РЅСЃРєСЂРёРїС†РёРё.
        
        callback(text: str) -> str (РёСЃРїСЂР°РІР»РµРЅРЅС‹Р№ С‚РµРєСЃС‚)
        """
        self._llm_fix_callback = callback
        kiwi_log("LLM", "LLM fix callback configured")
    
    def mute(self):
        """РћС‚РєР»СЋС‡Р°РµС‚ Р·Р°РїРёСЃСЊ РјРёРєСЂРѕС„РѕРЅР° (РІРѕ РІСЂРµРјСЏ TTS)."""
        self._is_muted = True
        kiwi_log("MUTE", "Microphone muted")
    
    def unmute(self):
        """Р’РєР»СЋС‡Р°РµС‚ Р·Р°РїРёСЃСЊ РјРёРєСЂРѕС„РѕРЅР° СЃ Р·Р°РґРµСЂР¶РєРѕР№ РґР»СЏ СЃС‚Р°Р±РёР»РёР·Р°С†РёРё."""
        # РќРµР±РѕР»СЊС€Р°СЏ Р·Р°РґРµСЂР¶РєР° С‡С‚РѕР±С‹ TTS СЌС…Рѕ РїРѕР»РЅРѕСЃС‚СЊСЋ Р·Р°С‚РёС…Р»Рѕ
        time.sleep(0.5)
        self._is_muted = False
        kiwi_log("MUTE", "Microphone unmuted")
    
    def _is_phantom_text(self, text: str) -> bool:
        """РџСЂРѕРІРµСЂСЏРµС‚, СЏРІР»СЏРµС‚СЃСЏ Р»Рё С‚РµРєСЃС‚ С„Р°РЅС‚РѕРјРЅС‹Рј (РіР°Р»Р»СЋС†РёРЅР°С†РёРµР№ Whisper)."""
        text_lower = text.lower().strip()
        
        # 1. РџСЂРѕРІРµСЂРєР° РїРѕ СЃРїРёСЃРєСѓ С„Р°РЅС‚РѕРјРЅС‹С… С„СЂР°Р·
        for phrase in PHANTOM_PHRASES:
            if phrase in text_lower:
                kiwi_log("PHANTOM", f"Filtered: '{text}' (contains '{phrase}')")
                return True
        
        # 2. РџСЂРѕРІРµСЂРєР° РїРѕ РїР°С‚С‚РµСЂРЅР°Рј С„Р°РЅС‚РѕРјРЅС‹С… Р·РІСѓРєРѕРІ
        for pattern in PHANTOM_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                kiwi_log("PHANTOM", f"Filtered: '{text}' (matches pattern '{pattern}')")
                return True
        
        return False
    
    def load_model(self):
        """Р—Р°РіСЂСѓР¶Р°РµС‚ РјРѕРґРµР»СЊ Whisper."""
        log_func = kiwi_log if UTILS_AVAILABLE else lambda tag, msg: print(f"[{tag}] {msg}", flush=True)
        
        log_func("WHISPER", f"load_model() called, model is None: {self.model is None}")
        if self.model is None:
            log_func("WHISPER", f"Loading model: {self.config.model_name}...")
            # РСЃРїРѕР»СЊР·СѓРµРј compute_type РёР· РєРѕРЅС„РёРіР° (РµСЃР»Рё Р·Р°РґР°РЅ), РёРЅР°С‡Рµ РІС‹С‡РёСЃР»СЏРµРј РїРѕ device
            compute_type = self.config.compute_type
            log_func("WHISPER", f"Using compute_type={compute_type}, device={self.config.device}")
            try:
                self.model = WhisperModel(
                    self.config.model_name,
                    device=self.config.device,
                    compute_type=compute_type,
                )
                log_func("WHISPER", "Model loaded successfully")
            except Exception as e:
                log_func("WHISPER", f"Failed to load model: {e}", level="ERROR")
                raise
    
    def start(self):
        """Р—Р°РїСѓСЃРєР°РµС‚ РїСЂРѕСЃР»СѓС€РёРІР°РЅРёРµ."""
        if self.is_running:
            return
        
        kiwi_log("LISTENER", "start() called")
        self.load_model()
        kiwi_log("LISTENER", "Model loaded, calibrating...")
        self._calibrate_noise_floor()  # РљР°Р»РёР±СЂСѓРµРј РїРѕСЂРѕРі С€СѓРјР°
        kiwi_log("LISTENER", "Calibration done")
        
        # Р—Р°РіСЂСѓР¶Р°РµРј VAD РїСЂРё СЃС‚Р°СЂС‚Рµ (РЅРµ Р»РµРЅРёРІРѕ) - С„РёРєСЃ РґР»СЏ "noVAD" РїСЂРѕР±Р»РµРјС‹
        kiwi_log("LISTENER", "Loading VAD...")
        self._ensure_vad_loaded()
        if self._vad_enabled:
            kiwi_log("LISTENER", f"VAD loaded successfully (enabled={self._vad_enabled})")
            self._reset_vad_state()  # РЎР±СЂР°СЃС‹РІР°РµРј РЅР°С‡Р°Р»СЊРЅРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ
        else:
            kiwi_log("LISTENER", "VAD not available, will use energy-based fallback", level="WARNING")
        
        self.is_running = True
        self._streaming_stop_event.clear()
        
        # РџРѕС‚РѕРє Р·Р°РїРёСЃРё Р°СѓРґРёРѕ
        self._recording_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._recording_thread.start()
        
        # РџРѕС‚РѕРє РѕР±СЂР°Р±РѕС‚РєРё Р°СѓРґРёРѕ
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()
        
        # РџРѕС‚РѕРє РѕС‚СЃР»РµР¶РёРІР°РЅРёСЏ С‚Р°Р№РјР°СѓС‚Р° РґРёР°Р»РѕРіР°
        self._dialog_timeout_thread = threading.Thread(target=self._dialog_timeout_loop, daemon=True)
        self._dialog_timeout_thread.start()
        
        # РџРѕС‚РѕРє СЃС‚СЂРёРјРёРЅРіРѕРІРѕР№ С‚СЂР°РЅСЃРєСЂРёРїС†РёРё (РµСЃР»Рё РІРєР»СЋС‡РµРЅР°)
        if self._streaming_enabled:
            self._streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self._streaming_thread.start()
        
        kiwi_log("MIC", f"Kiwi Listener started. Noise floor: {self._noise_floor:.6f}, threshold: {self._silence_threshold:.6f}")
    
    def _dialog_timeout_loop(self):
        """РћС‚СЃР»РµР¶РёРІР°РµС‚ С‚Р°Р№РјР°СѓС‚ СЂРµР¶РёРјР° РґРёР°Р»РѕРіР° Рё РІРѕСЃРїСЂРѕРёР·РІРѕРґРёС‚ idle Р·РІСѓРє.
        
        РўР°Р№РјР°СѓС‚ РЅРµ СЃСЂР°Р±Р°С‚С‹РІР°РµС‚ РµСЃР»Рё Kiwi СЃРµР№С‡Р°СЃ РіРѕРІРѕСЂРёС‚ (TTS).
        """
        while self.is_running:
            time.sleep(0.5)  # РџСЂРѕРІРµСЂСЏРµРј РєР°Р¶РґС‹Рµ 500 РјСЃ
            
            if self.dialog_mode:
                # === BARGE-IN: Р‘Р»РѕРєРёСЂСѓРµРј С‚Р°Р№РјР°СѓС‚ РµСЃР»Рё Kiwi РіРѕРІРѕСЂРёС‚ ===
                if self._is_kiwi_speaking():
                    # Kiwi РіРѕРІРѕСЂРёС‚ - РѕР±РЅРѕРІР»СЏРµРј С‚Р°Р№РјР°СѓС‚ С‡С‚РѕР±С‹ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»
                    self._extend_dialog_timeout()
                    continue
                
                remaining = self.dialog_until - time.time()
                
                # Р•СЃР»Рё С‚Р°Р№РјР°СѓС‚ РёСЃС‚С‘Рє - РІРѕР·РІСЂР°С‰Р°РµРјСЃСЏ РІ СЂРµР¶РёРј wake word
                if remaining <= 0 and not self._idle_played:
                    self._idle_played = True
                    self.dialog_mode = False
                    kiwi_log("DIALOG", "Timeout - back to wake word mode")
                    # Idle Р·РІСѓРє РїСЂРѕРёРіСЂС‹РІР°РµС‚СЃСЏ РѕС‚РґРµР»СЊРЅРѕ С‡РµСЂРµР· idle timer РІ kiwi_service

    def _calibrate_noise_floor(self):
        """РљР°Р»РёР±СЂСѓРµС‚ СѓСЂРѕРІРµРЅСЊ С€СѓРјР° РѕРєСЂСѓР¶РµРЅРёСЏ РїСЂРё СЃС‚Р°СЂС‚Рµ."""
        kiwi_log("CALIB", f"Calibrating noise floor ({self.config.noise_sample_duration}s)...")

        chunk_samples = int(self.config.chunk_duration * self.config.sample_rate)
        noise_samples = []
        
        def calibration_callback(indata, frames, time_info, status):
            audio_chunk = indata[:, 0].copy()
            volume = np.abs(audio_chunk).mean()
            noise_samples.append(volume)
        
        try:
            kiwi_log("CALIB", f"Opening InputStream (rate={self.config.sample_rate}, blocksize={chunk_samples})...")
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=chunk_samples,
                callback=calibration_callback
            ):
                kiwi_log("CALIB", f"Stream opened, sleeping {self.config.noise_sample_duration}s...")
                time.sleep(self.config.noise_sample_duration)
                kiwi_log("CALIB", "Sleep done, closing stream...")
        except Exception as e:
            kiwi_log("CALIB", f"Error: {e}", level="ERROR")
            self._noise_floor = 0.005
            self._silence_threshold = 0.015
            return
        
        if noise_samples:
            # РњРµРґРёР°РЅР° Р±РѕР»РµРµ СѓСЃС‚РѕР№С‡РёРІР° Рє РІС‹Р±СЂРѕСЃР°Рј
            self._noise_floor = np.median(noise_samples)
            self._silence_threshold = self._noise_floor * self.config.noise_threshold_multiplier
            # РњРёРЅРёРјР°Р»СЊРЅС‹Р№ РїРѕСЂРѕРі РґР»СЏ Р·Р°С‰РёС‚С‹ РѕС‚ С€СѓРјРѕРІ
            self._silence_threshold = max(self._silence_threshold, 0.008)
            kiwi_log("CALIB", f"Done: noise_floor={self._noise_floor:.6f}, threshold={self._silence_threshold:.6f}")
        else:
            self._noise_floor = 0.005
            self._silence_threshold = 0.015
            kiwi_log("CALIB", "No samples, using defaults", level="WARNING")
    
    def stop(self):
        """РћСЃС‚Р°РЅР°РІР»РёРІР°РµС‚ РїСЂРѕСЃР»СѓС€РёРІР°РЅРёРµ."""
        self.is_running = False
        self._streaming_stop_event.set()
        
        if self._recording_thread:
            self._recording_thread.join(timeout=2)
        if self._processing_thread:
            self._processing_thread.join(timeout=2)
        if self._streaming_thread:
            self._streaming_thread.join(timeout=2)
        
        kiwi_log("STOP", "Kiwi Listener stopped")
    
    def activate_dialog_mode(self):
        """РђРєС‚РёРІРёСЂСѓРµС‚ СЂРµР¶РёРј РґРёР°Р»РѕРіР° РЅР° dialog_timeout СЃРµРєСѓРЅРґ."""
        self.dialog_mode = True
        self.dialog_until = time.time() + self.dialog_timeout
        self._idle_played = False  # РЎР±СЂР°СЃС‹РІР°РµРј С„Р»Р°Рі idle Р·РІСѓРєР°
        kiwi_log("DIALOG", f"Activated for {self.dialog_timeout}s")

    def _extend_dialog_timeout(self, timeout: Optional[float] = None):
        """Extend dialog timeout without shortening an already longer deadline."""
        if not self.dialog_mode:
            return

        extension = self.dialog_timeout if timeout is None else timeout
        new_until = time.time() + extension
        if new_until > self.dialog_until:
            self.dialog_until = new_until
    
    def _check_dialog_mode(self) -> bool:
        """РџСЂРѕРІРµСЂСЏРµС‚ Р°РєС‚РёРІРµРЅ Р»Рё СЂРµР¶РёРј РґРёР°Р»РѕРіР°."""
        if self.dialog_mode:
            if time.time() > self.dialog_until:
                self.dialog_mode = False
                kiwi_log("DIALOG", "Timeout - back to wake word mode")
                # Idle Р·РІСѓРє РїСЂРѕРёРіСЂС‹РІР°РµС‚СЃСЏ РѕС‚РґРµР»СЊРЅРѕ С‡РµСЂРµР· idle timer РІ kiwi_service
                return False
            return True
        return False
    
    def _is_owner_speaker(self, speaker_id: str) -> bool:
        if self.speaker_manager is not None:
            try:
                return self.speaker_manager.is_owner(speaker_id)
            except Exception:
                return speaker_id == "owner"
        return speaker_id == "owner"
    
    def _can_owner_skip_wake_word(self, meta: Dict[str, Any]) -> Tuple[bool, str]:
        """Allow wake-word bypass only for owner in quiet 1:1 context."""
        if not self._owner_handsfree_enabled:
            return False, "disabled"
        
        speaker_id = str(meta.get("speaker_id", "unknown"))
        if not self._is_owner_speaker(speaker_id):
            return False, "not_owner"
        
        if speaker_id == "self":
            return False, "self_audio"
        
        music_probability = float(meta.get("music_probability", 0.0))
        if self._music_filter_enabled and music_probability >= self._music_reject_threshold:
            return False, "music"
        
        if self._last_non_owner_activity_at > 0:
            guest_age = time.time() - self._last_non_owner_activity_at
            if guest_age < self._owner_handsfree_guest_cooldown:
                return False, "guest_recent"
        
        return True, "ok"
    
    def _is_kiwi_speaking(self) -> bool:
        """РџСЂРѕРІРµСЂСЏРµС‚, РіРѕРІРѕСЂРёС‚ Р»Рё СЃРµР№С‡Р°СЃ Kiwi (РґР»СЏ barge-in Рё mute)."""
        if hasattr(self.on_wake_word, '__self__'):
            service = self.on_wake_word.__self__
            if hasattr(service, 'is_speaking'):
                return service.is_speaking()
        return False
    
    def _request_barge_in(self):
        """Р—Р°РїСЂР°С€РёРІР°РµС‚ РїСЂРµСЂС‹РІР°РЅРёРµ TTS РµСЃР»Рё Kiwi РіРѕРІРѕСЂРёС‚."""
        if hasattr(self.on_wake_word, '__self__'):
            service = self.on_wake_word.__self__
            if hasattr(service, 'request_barge_in'):
                service.request_barge_in()
    
    def _ensure_vad_loaded(self):
        """Р—Р°РіСЂСѓР¶Р°РµС‚ Silero VAD РјРѕРґРµР»СЊ (РµСЃР»Рё РµС‰Рµ РЅРµ Р·Р°РіСЂСѓР¶РµРЅР°)."""
        if self._vad_loaded or not TORCH_AVAILABLE:
            return
        
        try:
            kiwi_log("VAD", "Loading Silero VAD (first use)...")
            self._vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=True,  # ONNX Р±С‹СЃС‚СЂРµРµ РґР»СЏ CPU
            )
            self._vad_enabled = True
            self._vad_loaded = True
            kiwi_log("VAD", "Silero VAD loaded (ONNX mode)")
        except Exception as e:
            kiwi_log("VAD", f"Silero VAD not available: {e}", level="WARNING")
            kiwi_log("VAD", "Falling back to energy-only barge-in (consecutive chunks + raised threshold)", level="WARNING")
            self._vad_loaded = True

    def _reset_vad_state(self):
        """РЎР±СЂР°СЃС‹РІР°РµС‚ РІРЅСѓС‚СЂРµРЅРЅРµРµ СЃРѕСЃС‚РѕСЏРЅРёРµ Silero VAD (hidden state).
        
        Р­С‚Рѕ РІР°Р¶РЅРѕ РїРѕС‚РѕРјСѓ С‡С‚Рѕ Silero VAD вЂ” RNN, Рё РµРіРѕ hidden state РЅР°РєР°РїР»РёРІР°РµС‚СЃСЏ.
        Р‘РµР· СЃР±СЂРѕСЃР° СЃРѕСЃС‚РѕСЏРЅРёРµ "Р·Р°РіСЂСЏР·РЅСЏРµС‚СЃСЏ" Рё VAD РїРµСЂРµСЃС‚Р°С‘С‚ СЂР°Р±РѕС‚Р°С‚СЊ РєРѕСЂСЂРµРєС‚РЅРѕ.
        """
        if not self._vad_enabled or self._vad_model is None:
            return
        
        try:
            # Silero VAD РёРјРµРµС‚ РјРµС‚РѕРґ reset_states() РґР»СЏ СЃР±СЂРѕСЃР° hidden state
            if hasattr(self._vad_model, 'reset_states'):
                self._vad_model.reset_states()
                kiwi_log("VAD", "State reset (hidden state cleared)")
        except Exception as e:
            kiwi_log("VAD", f"Error resetting state: {e}", level="ERROR")
    
    def _check_vad(self, audio_chunk: np.ndarray) -> bool:
        """РџСЂРѕРІРµСЂСЏРµС‚ С‡Р°РЅРє Р°СѓРґРёРѕ С‡РµСЂРµР· Silero VAD вЂ” СЌС‚Рѕ СЂРµС‡СЊ РёР»Рё С€СѓРј?
        
        Silero VAD СЂР°Р±РѕС‚Р°РµС‚ ~1ms РЅР° С‡Р°РЅРє, РїРѕР»РЅРѕСЃС‚СЊСЋ Р»РѕРєР°Р»СЊРЅРѕ.
        Р•СЃР»Рё VAD РЅРµРґРѕСЃС‚СѓРїРµРЅ вЂ” fallback РЅР° True (РїСЂРѕРїСѓСЃРєР°РµРј РІСЃС‘).
        
        Args:
            audio_chunk: numpy float32 array, 16kHz
            
        Returns:
            True РµСЃР»Рё РѕР±РЅР°СЂСѓР¶РµРЅР° СЂРµС‡СЊ
        """
        # Р›РµРЅРёРІР°СЏ Р·Р°РіСЂСѓР·РєР° VAD РїСЂРё РїРµСЂРІРѕРј РІС‹Р·РѕРІРµ
        self._ensure_vad_loaded()
        
        if not self._vad_enabled or self._vad_model is None:
            return True  # Fallback: СЃС‡РёС‚Р°РµРј С‡С‚Рѕ СЌС‚Рѕ СЂРµС‡СЊ
        
        try:
            # Silero VAD РѕР¶РёРґР°РµС‚ torch tensor, 16kHz, mono
            audio_tensor = torch.from_numpy(audio_chunk).float()
            # VAD РѕР¶РёРґР°РµС‚ С‡Р°РЅРєРё РѕРїСЂРµРґРµР»С‘РЅРЅРѕР№ РґР»РёРЅС‹ (512 СЃРµРјРїР»РѕРІ РґР»СЏ 16kHz)
            # РќРѕ РјРѕР¶РµС‚ СЂР°Р±РѕС‚Р°С‚СЊ Рё СЃ РїСЂРѕРёР·РІРѕР»СЊРЅРѕР№ РґР»РёРЅРѕР№ РµСЃР»Рё > 512
            if len(audio_tensor) < 512:
                return True  # РЎР»РёС€РєРѕРј РєРѕСЂРѕС‚РєРёР№ С‡Р°РЅРє
            
            confidence = self._vad_model(audio_tensor, self.config.sample_rate).item()
            return confidence > 0.5  # РџРѕСЂРѕРі: >50% РІРµСЂРѕСЏС‚РЅРѕСЃС‚СЊ СЂРµС‡Рё
        except Exception:
            return True  # РџСЂРё РѕС€РёР±РєРµ вЂ” РїСЂРѕРїСѓСЃРєР°РµРј
    
    def _record_loop(self):
        """РќРµРїСЂРµСЂС‹РІРЅР°СЏ Р·Р°РїРёСЃСЊ Р°СѓРґРёРѕ СЃ РјРёРєСЂРѕС„РѕРЅР°.
        
        РСЃРїРѕР»СЊР·СѓРµС‚ Р°РґР°РїС‚РёРІРЅС‹Р№ РїРѕСЂРѕРі С€СѓРјР° РґР»СЏ С‚РѕС‡РЅРѕРіРѕ РѕРїСЂРµРґРµР»РµРЅРёСЏ РЅР°С‡Р°Р»Р°/РєРѕРЅС†Р° СЂРµС‡Рё.
        Р—Р°С‰РёС‚Р° РѕС‚ С„Р°РЅС‚РѕРјРЅС‹С… Р·РІСѓРєРѕРІ С‡РµСЂРµР· MIN_SPEECH_VOLUME.
        РЈРјРЅС‹Р№ barge-in: Silero VAD + consecutive chunks + grace period.
        """
        chunk_samples = int(self.config.chunk_duration * self.config.sample_rate)
        
        # === PRE-BUFFER: С…СЂР°РЅРёРј Р°СѓРґРёРѕ Р”Р•РўР•РљРўРР РћР’РђРќРРЇ СЂРµС‡Рё ===
        pre_buffer_size = int(self.config.pre_buffer_duration / self.config.chunk_duration)
        pre_buffer = deque(maxlen=pre_buffer_size)
        
        audio_buffer = []
        is_speaking = False
        silence_counter = 0
        speech_start_time = None
        
        # === VAD CONTINUATION LIMITER ===
        vad_continuation_count = 0  # РЎС‡С'С‚С‡РёРє РїСЂРѕРґР»РµРЅРёР№ Р·Р°РїРёСЃРё С‡РµСЂРµР· VAD
        MAX_VAD_CONTINUATIONS = 3    # РњР°РєСЃРёРјСѓРј РїСЂРѕРґР»РµРЅРёР№ РїРѕРґСЂСЏРґ

        # === CONTINUOUS NOISE RECALIBRATION ===
        noise_recalib_samples = []   # Ambient samples for recalibration
        NOISE_RECALIB_INTERVAL = 100  # Every 100 quiet chunks (~30s at 0.3s/chunk)
        quiet_chunk_counter = 0

        def audio_callback(indata, frames, time_info, status):
            nonlocal audio_buffer, is_speaking, silence_counter, speech_start_time, pre_buffer, vad_continuation_count
            nonlocal noise_recalib_samples, quiet_chunk_counter
            
            if status:
                status_text = str(status)
                now = time.time()
                if now - self._last_audio_status_log >= 1.0:
                    kiwi_log("MIC", f"Audio status: {status_text}", level="WARNING")
                    self._last_audio_status_log = now
                
                if "input overflow" in status_text.lower():
                    self._input_overflow_count += 1
                    if self._input_overflow_count % 10 == 1:
                        kiwi_log("MIC", f"Input overflow detected ({self._input_overflow_count}) - resetting capture state", level="WARNING")
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0
                    speech_start_time = None
                    pre_buffer.clear()
                    return
            
            audio_chunk = indata[:, 0].copy()
            volume = np.abs(audio_chunk).mean()
            
            # === PRE-BUFFER: Р’СЃРµРіРґР° СЃРѕС…СЂР°РЅСЏРµРј Р°СѓРґРёРѕ (РґР°Р¶Рµ РєРѕРіРґР° С‚РёС…Рѕ) ===
            pre_buffer.append(audio_chunk)
            
            # === DEBUG: РџРѕРєР°Р·С‹РІР°РµРј СѓСЂРѕРІРµРЅСЊ Р·РІСѓРєР° РєР°Р¶РґС‹Рµ 30 С‡Р°РЅРєРѕРІ (~10s) ===
            self._debug_counter = getattr(self, '_debug_counter', 0) + 1
            if self._debug_counter % 30 == 0:
                try:
                    import sys
                    vad_str = "VAD" if self._vad_enabled else "noVAD"
                    bar_len = int(min(volume * 100, 50))
                    bar = "#" * bar_len + "-" * (50 - bar_len)
                    sys.stdout.write(f"\r[LEVEL] |{bar}| {volume:.4f} (thr={self._silence_threshold:.4f}) {vad_str} {'SPEAKING' if is_speaking else ''}      ")
                    sys.stdout.flush()
                except:
                    pass
            
            # === BARGE-IN: РџСЂРѕРІРµСЂСЏРµРј РЅРµ РіРѕРІРѕСЂРёС‚ Р»Рё СЃРµР№С‡Р°СЃ Kiwi ===
            is_kiwi_speaking = self._is_kiwi_speaking()
            
            # === Р—РђР©РРўРђ РћРў Р¤РђРќРўРћРњРќР«РҐ Р—Р’РЈРљРћР’ ===
            # РџСЂРѕРІРµСЂСЏРµРј РѕР±Р° РїРѕСЂРѕРіР°: Р°РґР°РїС‚РёРІРЅС‹Р№ Р РјРёРЅРёРјР°Р»СЊРЅС‹Р№
            effective_min_speech_volume = self._get_effective_min_speech_volume()
            is_sound = volume > self._silence_threshold and volume > effective_min_speech_volume

            # === VAD OVERRIDE: volume выше min но ниже noise threshold →
            # Silero VAD решает (защита от завышенного noise floor) ===
            if not is_sound and not is_speaking and self._vad_enabled:
                if volume > effective_min_speech_volume and self._check_vad(audio_chunk):
                    is_sound = True

            # =================================================================
            # Р“Р›РђР’РќРђРЇ Р—РђР©РРўРђ: РљРѕРіРґР° Kiwi РіРѕРІРѕСЂРёС‚ вЂ” РќР• Р·Р°РїРёСЃС‹РІР°РµРј Р°СѓРґРёРѕ РґР»СЏ Whisper
            # РћР±СЂР°Р±Р°С‚С‹РІР°РµРј РўРћР›Р¬РљРћ barge-in Р»РѕРіРёРєСѓ. РРЅР°С‡Рµ Kiwi СѓСЃР»С‹С€РёС‚ СЃРІРѕР№ РіРѕР»РѕСЃ
            # С‡РµСЂРµР· РєРѕР»РѕРЅРєРё, Whisper СЂР°СЃС€РёС„СЂСѓРµС‚ TTS СЌС…Рѕ, Рё Kiwi РѕС‚РІРµС‚РёС‚ СЃР°РјРѕР№ СЃРµР±Рµ.
            # =================================================================
            if is_kiwi_speaking:
                if is_sound:
                    # РўРѕР»СЊРєРѕ barge-in Р»РѕРіРёРєР°, Р±РµР· Р·Р°РїРёСЃРё РІ audio_buffer
                    time_since_tts = time.time() - self._tts_start_time
                    if time_since_tts >= self._barge_in_grace_period:
                        # РџРѕРІС‹С€РµРЅРЅС‹Р№ РїРѕСЂРѕРі + Р°Р±СЃРѕР»СЋС‚РЅС‹Р№ РјРёРЅРёРјСѓРј
                        barge_in_threshold = max(
                            self._silence_threshold * self._barge_in_volume_multiplier,
                            self._barge_in_min_volume
                        )
                        if volume > barge_in_threshold:
                            is_real_speech = self._check_vad(audio_chunk)
                            if is_real_speech:
                                self._barge_in_counter += 1
                                if self._barge_in_counter >= self._barge_in_chunks_required:
                                    kiwi_log("BARGE-IN", f"Confirmed! vol={volume:.4f}, threshold={barge_in_threshold:.4f}, consecutive={self._barge_in_counter}")
                                    self._request_barge_in()
                                    self._barge_in_counter = 0
                            else:
                                self._barge_in_counter = 0
                        else:
                            self._barge_in_counter = 0
                else:
                    self._barge_in_counter = 0
                
                # РЎР±СЂР°СЃС‹РІР°РµРј speech state вЂ” Р·Р°РїРёСЃСЊ Р°СѓРґРёРѕ РґР»СЏ Whisper РЅРµ РІРµРґС‘С‚СЃСЏ
                if is_speaking:
                    kiwi_log("MIC", "Speech recording stopped — Kiwi is speaking (echo protection)")
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0
                    speech_start_time = None
                return  # в†ђ Р’Р«РҐРћР”РРњ, РЅРµ Р·Р°РїРёСЃС‹РІР°РµРј Р°СѓРґРёРѕ
            
            # =================================================================
            # РњРЃР РўР’РђРЇ Р—РћРќРђ: 200РјСЃ РїРѕСЃР»Рµ РєРѕСЂРѕС‚РєРёС… Р·РІСѓРєРѕРІ (idle, beep, startup)
            # =================================================================
            time_since_sound = time.time() - self._sound_end_time
            if time_since_sound < self._post_sound_dead_zone:
                if is_speaking:
                    kiwi_log("MIC", f"Speech recording stopped — post-sound dead zone ({time_since_sound:.1f}s < {self._post_sound_dead_zone}s)")
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0
                    speech_start_time = None
                return
            
            # =================================================================
            # РњРЃР РўР’РђРЇ Р—РћРќРђ: 1СЃ РїРѕСЃР»Рµ Р·Р°РІРµСЂС€РµРЅРёСЏ TTS (СЌС…Рѕ РіРѕР»РѕСЃР° Kiwi Р·Р°С‚РёС…Р°РµС‚)
            # =================================================================
            time_since_tts = time.time() - self._tts_start_time
            if time_since_tts < self._post_tts_dead_zone:
                # Р­С…Рѕ РµС‰С‘ РјРѕР¶РµС‚ Р±С‹С‚СЊ РІ РєРѕРјРЅР°С‚Рµ вЂ” РЅРµ Р·Р°РїРёСЃС‹РІР°РµРј
                if is_speaking:
                    kiwi_log("MIC", f"Speech recording stopped — post-TTS dead zone ({time_since_tts:.1f}s < {self._post_tts_dead_zone}s)")
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0
                    speech_start_time = None
                return
            
            # =================================================================
            # РћР‘Р«Р§РќР«Р™ Р Р•Р–РРњ: Kiwi РќР• РіРѕРІРѕСЂРёС‚ вЂ” РЅРѕСЂРјР°Р»СЊРЅР°СЏ Р·Р°РїРёСЃСЊ СЂРµС‡Рё
            # =================================================================
            self._barge_in_counter = 0  # РЎР±СЂРѕСЃ barge-in (РЅРµ РЅСѓР¶РµРЅ РєРѕРіРґР° Kiwi РјРѕР»С‡РёС‚)
            
            if is_sound:
                # Р”РѕРїРѕР»РЅРёС‚РµР»СЊРЅР°СЏ РїСЂРѕРІРµСЂРєР° С‡РµСЂРµР· Silero VAD РїРµСЂРµРґ РЅР°С‡Р°Р»РѕРј Р·Р°РїРёСЃРё
                if not is_speaking and self._vad_enabled:
                    # РџСЂРѕРІРµСЂСЏРµРј РЅРµСЃРєРѕР»СЊРєРѕ РїРѕСЃР»РµРґРЅРёС… С‡Р°РЅРєРѕРІ С‡РµСЂРµР· VAD
                    vad_speech_frames = 0
                    for chunk in list(pre_buffer)[-3:]:  # РџРѕСЃР»РµРґРЅРёРµ 3 С‡Р°РЅРєР°
                        if self._check_vad(chunk):
                            vad_speech_frames += 1
                    # РќР°С‡РёРЅР°РµРј Р·Р°РїРёСЃСЊ С‚РѕР»СЊРєРѕ РµСЃР»Рё VAD СѓРІРёРґРµР» СЂРµС‡СЊ
                    if vad_speech_frames < 2:  # РњРёРЅРёРјСѓРј 2 РёР· 3 С‡Р°РЅРєРѕРІ РґРѕР»Р¶РЅС‹ Р±С‹С‚СЊ СЂРµС‡СЊСЋ
                        # РЎР±СЂР°СЃС‹РІР°РµРј pre_buffer С‡С‚РѕР±С‹ РЅРµ РЅР°РєР°РїР»РёРІР°С‚СЊ С€СѓРј
                        pre_buffer.clear()
                        return
                
                # Р›РѕРіРёСЂСѓРµРј С‚РѕР»СЊРєРѕ РЅР°С‡Р°Р»Рѕ СЂРµС‡Рё (РЅРµ РєР°Р¶РґС‹Р№ С‡Р°РЅРє)
                if not is_speaking:
                    is_speaking = True
                    speech_start_time = time.time()
                    # === PRE-BUFFER: РСЃРїРѕР»СЊР·СѓРµРј РЅР°РєРѕРїР»РµРЅРЅРѕРµ Р°СѓРґРёРѕ РёР· pre-buffer ===
                    audio_buffer = list(pre_buffer)
                    kiwi_log("MIC", f"Speech started: vol={volume:.4f}, threshold={self._silence_threshold:.4f}, min_vol={effective_min_speech_volume:.4f}, pre_buffer={len(pre_buffer)} chunks ({len(pre_buffer)*self.config.chunk_duration*1000:.0f}ms)")
                    
                    # === FIX: Р—Р°РїРѕРјРёРЅР°РµРј, Р±С‹Р»Р° Р»Рё СЂРµС‡СЊ РЅР°С‡Р°С‚Р° РІ СЂРµР¶РёРјРµ РґРёР°Р»РѕРіР° ===
                    self._speech_started_in_dialog = self.dialog_mode
                    if self._speech_started_in_dialog:
                        kiwi_log("DIALOG", "Speech started while in dialog mode (will process as command)")
                        # РџСЂРѕРґР»РµРІР°РµРј С‚Р°Р№РјР°СѓС‚ РЅР° РІСЂРµРјСЏ СЂРµС‡Рё
                        self._extend_dialog_timeout()
                    
                    # === РЎРўР РРњРРќР“: РРЅРёС†РёР°Р»РёР·РёСЂСѓРµРј СЃС‚СЂРёРјРµСЂ РїСЂРё РЅР°С‡Р°Р»Рµ СЂРµС‡Рё ===
                    if self._streaming_enabled and not self._early_wake_detected:
                        self._streaming_transcriber = StreamingTranscriber(
                            model=self.model,
                            sample_rate=self.config.sample_rate,
                            chunk_interval=self._streaming_chunk_interval,
                            min_audio_for_stream=self._streaming_min_audio
                        )
                else:
                    # FIX: РџСЂРѕРґР»РµРІР°РµРј С‚Р°Р№РјР°СѓС‚ РґРёР°Р»РѕРіР° РїРѕРєР° РёРґС‘С‚ СЂРµС‡СЊ (С‡С‚РѕР±С‹ РЅРµ РїСЂРµСЂРІР°С‚СЊ)
                    if self.dialog_mode:
                        self._extend_dialog_timeout()
                
                silence_counter = 0
                audio_buffer.append(audio_chunk)
                
                # === РЎРўР РРњРРќР“: РќР°РєР°РїР»РёРІР°РµРј Р°СѓРґРёРѕ РґР»СЏ partial transcribe ===
                # Р’РђР–РќРћ: С‚СЂР°РЅСЃРєСЂРёР±Р°С†РёСЏ РїСЂРѕРёСЃС…РѕРґРёС‚ РІ РћРўР”Р•Р›Р¬РќРћРњ РїРѕС‚РѕРєРµ (_streaming_loop)
                # audio_callback РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ РњРђРљРЎРРњРђР›Р¬РќРћ Р‘Р«РЎРўР Р«Рњ С‡С‚РѕР±С‹ РЅРµ Р±С‹Р»Рѕ overflow
                if self._streaming_enabled and self._streaming_transcriber is not None and not self._early_wake_detected:
                    self._streaming_transcriber.add_audio(audio_chunk)
                    # РќР• РІС‹Р·С‹РІР°РµРј transcribe_partial() Р·РґРµСЃСЊ - СЌС‚Рѕ Р±Р»РѕРєРёСЂСѓРµС‚ callback!
                
            else:
                if is_speaking:
                    silence_counter += 1
                    audio_buffer.append(audio_chunk)
                    
                    # === РђР”РђРџРўРР’РќРђРЇ РџРђРЈР—Рђ: РІС‹С‡РёСЃР»СЏРµРј РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ СЂРµС‡Рё ===
                    current_speech_duration = time.time() - speech_start_time if speech_start_time else 0
                    required_silence = self._get_silence_duration(current_speech_duration)
                    current_silence = silence_counter * self.config.chunk_duration
                    
                    if current_silence >= required_silence:
                        # === VAD-РџР РћР’Р•Р РљРђ: СѓР±РµР¶РґР°РµРјСЃСЏ С‡С‚Рѕ СЂРµС‡СЊ РґРµР№СЃС‚РІРёС‚РµР»СЊРЅРѕ Р·Р°РєРѕРЅС‡РёР»Р°СЃСЊ ===
                        
                        # РџСЂРѕРІРµСЂСЏРµРј СЃСЂРµРґРЅСЋСЋ РіСЂРѕРјРєРѕСЃС‚СЊ РїРѕСЃР»РµРґРЅРёС… С‡Р°РЅРєРѕРІ
                        # Р•СЃР»Рё РіСЂРѕРјРєРѕСЃС‚СЊ СЃС‚Р°Р±РёР»СЊРЅРѕ РЅРёР¶Рµ РїРѕСЂРѕРіР° вЂ” СЌС‚Рѕ С‚РѕС‡РЅРѕ С‚РёС€РёРЅР°, VAD РЅРµ РЅСѓР¶РµРЅ
                        recent_chunks = audio_buffer[-self._vad_end_speech_frames:] if len(audio_buffer) >= self._vad_end_speech_frames else audio_buffer
                        recent_volumes = [np.abs(c).mean() for c in recent_chunks]
                        avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 0
                        
                        # === РЈР›РЈР§РЁР•РќРќРђРЇ Р›РћР“РРљРђ: РіСЂРѕРјРєРѕСЃС‚СЊ + VAD + Р»РёРјРёС‚ ===
                        should_extend = False
                        
                        if avg_recent_volume >= effective_min_speech_volume:
                            # Громкость выше минимума речи — проверяем VAD
                            if self._check_vad_continuation(audio_buffer):
                                vad_continuation_count += 1
                                if vad_continuation_count <= MAX_VAD_CONTINUATIONS:
                                    should_extend = True
                                else:
                                    kiwi_log("VAD", f"Max continuations ({MAX_VAD_CONTINUATIONS}) reached, forcing end")
                        
                        if should_extend:
                            # VAD РІРёРґРёС‚ РїСЂРѕРґРѕР»Р¶РµРЅРёРµ СЂРµС‡Рё вЂ” РґРѕР±Р°РІР»СЏРµРј Р±РѕРЅСѓСЃ Рє РїР°СѓР·Рµ
                            silence_counter = max(0, silence_counter - self._vad_continuation_bonus_chunks)
                            kiwi_log("VAD", f"Continuation detected ({vad_continuation_count}/{MAX_VAD_CONTINUATIONS}), extending recording (bonus={self._vad_continuation_bonus_chunks} chunks)")
                        else:
                            # Р—Р°РІРµСЂС€Р°РµРј Р·Р°РїРёСЃСЊ
                            vad_continuation_count = 0  # РЎР±СЂР°СЃС‹РІР°РµРј СЃС‡С‘С‚С‡РёРє
                            duration = len(audio_buffer) * self.config.chunk_duration
                            if duration >= self.config.min_speech_duration:
                                kiwi_log("END", f"Speech ended, duration: {duration:.1f}s (silence: {current_silence:.1f}s >= {required_silence:.1f}s)")
                                self._submit_audio(audio_buffer.copy())
                                # РЎР±СЂР°СЃС‹РІР°РµРј VAD РїРѕСЃР»Рµ РѕС‚РїСЂР°РІРєРё utterance РІ РѕР±СЂР°Р±РѕС‚РєСѓ
                                self._reset_vad_state()
                            
                            audio_buffer = []
                            is_speaking = False
                            silence_counter = 0
                            speech_start_time = None
                
                # === CONTINUOUS NOISE RECALIBRATION ===
                # When idle (not speaking, not Kiwi speaking), collect ambient samples
                # and periodically recalibrate the noise floor
                if not is_speaking and not is_kiwi_speaking:
                    noise_recalib_samples.append(volume)
                    quiet_chunk_counter += 1
                    if quiet_chunk_counter >= NOISE_RECALIB_INTERVAL:
                        if noise_recalib_samples:
                            new_floor = np.median(noise_recalib_samples)
                            new_threshold = max(new_floor * self.config.noise_threshold_multiplier, 0.005)
                            if abs(new_threshold - self._silence_threshold) > 0.002:
                                kiwi_log("CALIB", f"Recalibrated: noise_floor={new_floor:.6f}, threshold={self._silence_threshold:.4f} -> {new_threshold:.4f}")
                                self._noise_floor = new_floor
                                self._silence_threshold = new_threshold
                        noise_recalib_samples.clear()
                        quiet_chunk_counter = 0
                else:
                    # Reset recalibration counter when speaking
                    noise_recalib_samples.clear()
                    quiet_chunk_counter = 0

                if speech_start_time and (time.time() - speech_start_time > self._max_speech_duration):
                    kiwi_log("END", "Max speech duration reached")
                    self._submit_audio(audio_buffer.copy())
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0
                    speech_start_time = None
        
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples,
            callback=audio_callback
        ):
            while self.is_running:
                time.sleep(0.1)
    
    def drain_audio_queue(self):
        """РћС‡РёС‰Р°РµС‚ РѕС‡РµСЂРµРґСЊ Р°СѓРґРёРѕ вЂ” РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ РїСЂРё РїРµСЂРµС…РѕРґР°С… СЃРѕСЃС‚РѕСЏРЅРёР№."""
        cleared = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared > 0:
            kiwi_log("QUEUE", f"Drained {cleared} audio items from queue")
    
    def _reset_streaming_state(self):
        """РЎР±СЂР°СЃС‹РІР°РµС‚ СЃРѕСЃС‚РѕСЏРЅРёРµ СЃС‚СЂРёРјРёРЅРіРѕРІРѕР№ С‚СЂР°РЅСЃРєСЂРёРїС†РёРё (thread-safe)."""
        with self._early_wake_lock:
            self._early_wake_detected = False
            self._early_command = ""
            self._early_detected_at = 0.0
        if self._streaming_transcriber is not None:
            self._streaming_transcriber.clear()
            self._streaming_transcriber = None
    
    def _streaming_loop(self):
        """
        РћС‚РґРµР»СЊРЅС‹Р№ РїРѕС‚РѕРє РґР»СЏ СЃС‚СЂРёРјРёРЅРіРѕРІРѕР№ С‚СЂР°РЅСЃРєСЂРёРїС†РёРё.
        РќР• Р±Р»РѕРєРёСЂСѓРµС‚ audio callback - СЂР°Р±РѕС‚Р°РµС‚ РїР°СЂР°Р»Р»РµР»СЊРЅРѕ.
        """
        kiwi_log("STREAMING", "Thread started")
        while self.is_running and not self._streaming_stop_event.is_set():
            try:
                # РџСЂРѕРІРµСЂСЏРµРј, РµСЃС‚СЊ Р»Рё Р°РєС‚РёРІРЅС‹Р№ СЃС‚СЂРёРјРµСЂ Рё РЅСѓР¶РЅРѕ Р»Рё С‚СЂР°РЅСЃРєСЂРёР±РёСЂРѕРІР°С‚СЊ
                if (self._streaming_enabled and 
                    self._streaming_transcriber is not None):
                    
                    # РџСЂРѕРІРµСЂСЏРµРј, РЅРµ РѕР±РЅР°СЂСѓР¶РµРЅ Р»Рё СѓР¶Рµ early wake
                    with self._early_wake_lock:
                        early_already_detected = self._early_wake_detected
                    
                    if not early_already_detected and self._streaming_transcriber.should_transcribe():
                        # Р—Р°РїСѓСЃРєР°РµРј С‚СЂР°РЅСЃРєСЂРёР±Р°С†РёСЋ РІ РѕС‚РґРµР»СЊРЅРѕРј РїРѕС‚РѕРєРµ (РЅРµ Р±Р»РѕРєРёСЂСѓРµРј СЌС‚РѕС‚ С†РёРєР»)
                        partial_text = self._streaming_transcriber.transcribe_partial(
                            fix_callback=self._fix_transcription
                        )
                        
                        if partial_text:
                            kiwi_log("STREAM", f"Partial: '{partial_text}'")
                            
                            # РџСЂРѕРІРµСЂСЏРµРј wake word РІ partial С‚РµРєСЃС‚Рµ
                            is_address, command = self.detector.is_direct_address(partial_text)
                            
                            if is_address and command:
                                # EARLY WAKE DETECTED! (thread-safe)
                                with self._early_wake_lock:
                                    if not self._early_wake_detected:  # Р”РІРѕР№РЅР°СЏ РїСЂРѕРІРµСЂРєР°
                                        kiwi_log("STREAM", f"Early wake word detected: '{command}'")
                                        self._early_wake_detected = True
                                        self._early_command = command
                                        self._early_detected_at = time.time()
                
                # РЎРїРёРј РЅРµРјРЅРѕРіРѕ С‡С‚РѕР±С‹ РЅРµ РіСЂСѓР·РёС‚СЊ CPU
                time.sleep(0.1)
                
            except Exception as e:
                kiwi_log("STREAMING", f"Error in streaming loop: {e}", level="ERROR")
                time.sleep(0.5)  # РџСЂРё РѕС€РёР±РєРµ Р¶РґС‘Рј РґРѕР»СЊС€Рµ
        
        kiwi_log("STREAMING", "Thread stopped")
    
    def _submit_audio(self, audio_chunks: list):
        """РћС‚РїСЂР°РІР»СЏРµС‚ Р°СѓРґРёРѕ РІ РѕС‡РµСЂРµРґСЊ РЅР° РѕР±СЂР°Р±РѕС‚РєСѓ СЃ РїСЂРѕРІРµСЂРєРѕР№ speaker ID Рё РґРµР±Р°СѓРЅСЃРёРЅРіРѕРј."""
        if not audio_chunks:
            return
        
        # === Р”Р•Р‘РђРЈРќРЎРРќР“: РќРµ РѕС‚РїСЂР°РІР»СЏС‚СЊ СЃР»РёС€РєРѕРј С‡Р°СЃС‚Рѕ ===
        current_time = time.time()
        if current_time - self._last_submit_time < self._submit_debounce:
            kiwi_log("SUBMIT", f"Debounced: too soon ({current_time - self._last_submit_time:.2f}s < {self._submit_debounce}s)")
            return
        self._last_submit_time = current_time
        
        # === РЎРўР РРњРРќР“: РЎР±СЂР°СЃС‹РІР°РµРј СЃС‚СЂРёРјРёРЅРіРѕРІРѕРµ СЃРѕСЃС‚РѕСЏРЅРёРµ РїСЂРё submit ===
        if self._streaming_enabled:
            self._reset_streaming_state()
        
        audio = np.concatenate(audio_chunks)
        duration = len(audio) / self.config.sample_rate

        # === ENERGY GATE: РѕС‚СЃРµРёРІР°РµРј Р±СѓС„РµСЂС‹ СЃ РЅРёР·РєРѕР№ СЌРЅРµСЂРіРёРµР№ (С„РѕРЅРѕРІС‹Р№ С€СѓРј) ===
        # Порог = silence_threshold (уже откалиброван выше noise floor).
        # Не умножаем — RMS буфера включает pre-buffer и trailing silence,
        # поэтому всегда ниже пиковой громкости отдельных чанков.
        rms = np.sqrt(np.mean(audio ** 2))
        energy_gate_threshold = 0.006  # fixed low gate; real filtering: VAD + noisereduce + Whisper
        if rms < energy_gate_threshold:
            kiwi_log("SUBMIT", f"Rejected: energy gate (rms={rms:.4f} < {energy_gate_threshold:.4f}). Likely noise.")
            return

        # === NOISE REDUCTION: РѕС‡РёСЃС‚РєР° Р°СѓРґРёРѕ РѕС‚ СЃС‚Р°С†РёРѕРЅР°СЂРЅРѕРіРѕ С€СѓРјР° ===
        if NOISEREDUCE_AVAILABLE:
            try:
                rms_before = rms
                audio = nr.reduce_noise(
                    y=audio,
                    sr=self.config.sample_rate,
                    stationary=True,
                    prop_decrease=0.4,
                )
                rms_after = np.sqrt(np.mean(audio ** 2))
                kiwi_log("DENOISE", f"Noise reduction applied (rms: {rms_before:.4f} -> {rms_after:.4f})")
            except Exception as e:
                kiwi_log("DENOISE", f"Noise reduction error: {e}", level="WARNING")

        # === VAD CHECK: Р¤РёРЅР°Р»СЊРЅР°СЏ РїСЂРѕРІРµСЂРєР° С‡РµСЂРµР· Silero VAD РїРµСЂРµРґ РѕС‚РїСЂР°РІРєРѕР№ РІ Whisper ===
        # Р Р°Р·Р±РёРІР°РµРј Р°СѓРґРёРѕ РЅР° С‡Р°РЅРєРё Рё РїСЂРѕРІРµСЂСЏРµРј С‡С‚Рѕ РІ РЅС‘Рј РґРµР№СЃС‚РІРёС‚РµР»СЊРЅРѕ РµСЃС‚СЊ СЂРµС‡СЊ
        if self._vad_enabled and duration >= 0.5:
            try:
                chunk_size = int(0.03 * self.config.sample_rate)  # 30ms chunks
                vad_speech_frames = 0
                total_vad_frames = 0
                
                for i in range(0, len(audio) - chunk_size, chunk_size):
                    chunk = audio[i:i + chunk_size]
                    if len(chunk) >= 512:  # РњРёРЅРёРјР°Р»СЊРЅС‹Р№ СЂР°Р·РјРµСЂ РґР»СЏ VAD
                        total_vad_frames += 1
                        if self._check_vad(chunk):
                            vad_speech_frames += 1
                
                # Р•СЃР»Рё РјРµРЅРµРµ 20% РєР°РґСЂРѕРІ СЃРѕРґРµСЂР¶Р°С‚ СЂРµС‡СЊ вЂ” СЌС‚Рѕ СЃРєРѕСЂРµРµ РІСЃРµРіРѕ С€СѓРј
                if total_vad_frames > 0 and vad_speech_frames / total_vad_frames < 0.2:
                    kiwi_log("SUBMIT", f"Rejected: VAD detected only {vad_speech_frames}/{total_vad_frames} speech frames ({vad_speech_frames/total_vad_frames*100:.1f}%). Likely noise.")
                    return
                    
            except Exception as e:
                kiwi_log("SUBMIT", f"VAD check error: {e}", level="ERROR")
        
        kiwi_log("SUBMIT", f"Submitting {duration:.1f}s audio to queue")

        # === SPEAKER IDENTIFICATION + MUSIC FILTER ===
        speaker_id = "unknown"
        speaker_name = "РќРµР·РЅР°РєРѕРјРµС†"
        speaker_priority = int(VoicePriority.GUEST) if SPEAKER_MANAGER_AVAILABLE else 2
        speaker_conf = 0.0

        # 1) Р—Р°С‰РёС‚Р° РѕС‚ self-echo
        if self.speaker_id is not None and duration >= 0.5:
            try:
                is_self, confidence = self.speaker_id.is_self_speaking(audio, self.config.sample_rate)
                if is_self:
                    kiwi_log("SPEAKER", f"Detected SELF voice (echo), ignoring. Confidence: {confidence:.2f}")
                    return
            except Exception as e:
                kiwi_log("SPEAKER", f"Self-check error: {e}", level="ERROR")

        # 2) Р‘С‹СЃС‚СЂР°СЏ РёРґРµРЅС‚РёС„РёРєР°С†РёСЏ РіРѕРІРѕСЂСЏС‰РµРіРѕ (owner/friend/guest)
        if duration >= 0.5:
            try:
                sid, prio, conf, name = self.identify_speaker_fast(audio)
                speaker_id = sid or "unknown"
                speaker_priority = int(prio)
                speaker_conf = float(conf)
                speaker_name = name or speaker_id
                if speaker_id != "unknown":
                    kiwi_log("SPEAKER", f"Detected: {speaker_name} ({speaker_id}), conf={speaker_conf:.2f}")
            except Exception as e:
                kiwi_log("SPEAKER", f"Fast identification error: {e}", level="ERROR")

        # 3) Р¤РѕРЅРѕРІР°СЏ РјСѓР·С‹РєР°: СЂРµР¶РµРј С‚РѕР»СЊРєРѕ СЏРІРЅС‹Рµ РјСѓР·С‹РєР°Р»СЊРЅС‹Рµ С„СЂР°РіРјРµРЅС‚С‹ Р±РµР· РёР·РІРµСЃС‚РЅРѕРіРѕ РіРѕР»РѕСЃР°
        music_probability = self._estimate_music_probability(audio)
        if (
            self._music_filter_enabled
            and music_probability >= self._music_reject_threshold
            and speaker_id in ("unknown", "self")
        ):
            kiwi_log("MUSIC", f"Rejected likely music chunk (prob={music_probability:.2f}, speaker={speaker_id})")
            return

        meta = {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "priority": speaker_priority,
            "confidence": speaker_conf,
            "music_probability": music_probability,
            "timestamp": time.time(),
        }

        self.audio_queue.put((audio, meta))
    
    # === VOICE PRIORITY QUEUE METHODS ===
    
    def identify_speaker_fast(self, audio: np.ndarray) -> Tuple[str, int, float, str]:
        """
        Р‘С‹СЃС‚СЂР°СЏ РёРґРµРЅС‚РёС„РёРєР°С†РёСЏ РіРѕРІРѕСЂСЏС‰РµРіРѕ С‡РµСЂРµР· Speaker Manager.
        
        Returns:
            (speaker_id, priority, confidence, display_name)
        """
        if self.speaker_manager is not None:
            return self.speaker_manager.identify_speaker_fast(
                audio, self.config.sample_rate
            )
        
        # Fallback - РёСЃРїРѕР»СЊР·СѓРµРј Р±Р°Р·РѕРІС‹Р№ speaker_id
        if self.speaker_id is not None:
            speaker_id, confidence = self.speaker_id.identify_speaker(
                audio, self.config.sample_rate
            )
            priority = 2  # GUEST
            return speaker_id, priority, confidence, speaker_id
        
        return "unknown", 2, 0.0, "РќРµР·РЅР°РєРѕРјРµС†"
    
    def check_owner_override(self, speaker_id: str) -> bool:
        """
        РџСЂРѕРІРµСЂСЏРµС‚, СЏРІР»СЏРµС‚СЃСЏ Р»Рё СЌС‚Рѕ OWNER Рё РЅСѓР¶РЅРѕ Р»Рё РїСЂРµСЂРІР°С‚СЊ С‚РµРєСѓС‰СѓСЋ Р·Р°РґР°С‡Сѓ.
        
        Returns:
            True РµСЃР»Рё СЌС‚Рѕ OWNER Рё РµСЃС‚СЊ Р°РєС‚РёРІРЅР°СЏ Р·Р°РґР°С‡Р°
        """
        if self.speaker_manager is not None:
            return self.speaker_manager.is_owner(speaker_id)
        return speaker_id == "owner"
    
    def get_last_speaker_id(self) -> Optional[str]:
        """Р’РѕР·РІСЂР°С‰Р°РµС‚ ID РїРѕСЃР»РµРґРЅРµРіРѕ РіРѕРІРѕСЂСЏС‰РµРіРѕ."""
        if self.speaker_manager is not None:
            return self.speaker_manager.voice_context.speaker_id
        return None
    
    def update_voice_context(self, speaker_id: str, speaker_name: str, priority: int, confidence: float, command: str = ""):
        """РћР±РЅРѕРІР»СЏРµС‚ РєРѕРЅС‚РµРєСЃС‚ РіРѕР»РѕСЃР°."""
        if self.speaker_manager is not None:
            from kiwi.speaker_manager import VoicePriority
            self.speaker_manager.update_context(speaker_id, speaker_name, VoicePriority(priority), confidence, command)
    
    def block_current_speaker(self) -> bool:
        """Р‘Р»РѕРєРёСЂСѓРµС‚ РїРѕСЃР»РµРґРЅРµРіРѕ РіРѕРІРѕСЂСЏС‰РµРіРѕ."""
        if self.speaker_manager is not None:
            last_id = self.get_last_speaker_id()
            if last_id:
                return self.speaker_manager.block_speaker(last_id)
        return False
    
    def unblock_speaker_by_name(self, name: str) -> bool:
        """Р Р°Р·Р±Р»РѕРєРёСЂСѓРµС‚ РіРѕР»РѕСЃ РїРѕ РёРјРµРЅРё."""
        if self.speaker_manager is not None:
            # РС‰РµРј РїРѕ РёРјРµРЅРё РІ РїСЂРѕС„РёР»СЏС…
            for pid, profile in self.speaker_manager.profiles.items():
                if name.lower() in profile.display_name.lower():
                    return self.speaker_manager.unblock_speaker(pid)
        return False
    
    def handle_voice_control_command(self, command: str, audio: np.ndarray) -> Tuple[bool, str]:
        """
        РћР±СЂР°Р±Р°С‚С‹РІР°РµС‚ РєРѕРјР°РЅРґС‹ СѓРїСЂР°РІР»РµРЅРёСЏ РіРѕР»РѕСЃР°РјРё РѕС‚ OWNER.
        
        Returns:
            (handled, response)
        """
        if self.speaker_manager is None:
            return False, ""
        
        command_lower = command.lower()
        
        # РџСЂРѕРІРµСЂСЏРµРј РїР°С‚С‚РµСЂРЅС‹
        for pattern, action in OWNER_CONTROL_PATTERNS.items():
            if re.search(pattern, command_lower):
                kiwi_log("VOICE_CONTROL", f"Matched: {action} for '{command}'")
                
                if action == "block_last":
                    success = self.block_current_speaker()
                    return True, "Р“РѕР»РѕСЃ Р·Р°Р±Р»РѕРєРёСЂРѕРІР°РЅ." if success else "РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°Р±Р»РѕРєРёСЂРѕРІР°С‚СЊ."
                
                elif action == "unblock_last":
                    # РџС‹С‚Р°РµРјСЃСЏ СЂР°Р·Р±Р»РѕРєРёСЂРѕРІР°С‚СЊ РїРѕ РєРѕРЅС‚РµРєСЃС‚Сѓ
                    last_id = self.get_last_speaker_id()
                    if last_id:
                        success = self.speaker_manager.unblock_speaker(last_id)
                        return True, "Р“РѕР»РѕСЃ СЂР°Р·Р±Р»РѕРєРёСЂРѕРІР°РЅ." if success else "РќРµ СѓРґР°Р»РѕСЃСЊ СЂР°Р·Р±Р»РѕРєРёСЂРѕРІР°С‚СЊ."
                    return True, "РќРµС‚ РіРѕР»РѕСЃР° РґР»СЏ СЂР°Р·Р±Р»РѕРєРёСЂРѕРІРєРё."
                
                elif action == "add_friend":
                    name = extract_name_from_command(command)
                    if name:
                        success, sid = self.speaker_manager.add_friend(audio, self.config.sample_rate, name)
                        return True, f"Р—Р°РїРѕРјРЅРёР»Р° {name}!" if success else "РќРµ СѓРґР°Р»РѕСЃСЊ Р·Р°РїРѕРјРЅРёС‚СЊ."
                    return True, "РЎРєР°Р¶Рё РёРјСЏ РґР»СЏ Р·Р°РїРѕРјРёРЅР°РЅРёСЏ."
                
                elif action == "forget_speaker":
                    last_id = self.get_last_speaker_id()
                    if last_id:
                        if last_id in self.speaker_manager.profiles:
                            name = self.speaker_manager.profiles[last_id].display_name
                            del self.speaker_manager.profiles[last_id]
                            self.speaker_manager._save_extended_profiles()
                            return True, f"Р—Р°Р±С‹Р»Р° {name}."
                    return True, "РќРµС‚ РіРѕР»РѕСЃР° РґР»СЏ СѓРґР°Р»РµРЅРёСЏ."
                
                elif action == "identify":
                    if audio is not None:
                        response = self.speaker_manager.who_am_i(audio, self.config.sample_rate)
                        return True, response
                    return True, "РќРµС‚ Р°СѓРґРёРѕ РґР»СЏ РёРґРµРЅС‚РёС„РёРєР°С†РёРё."
                
                elif action == "list_voices":
                    profiles = self.speaker_manager.get_profile_info()
                    if profiles:
                        lines = ["РР·РІРµСЃС‚РЅС‹Рµ РіРѕР»РѕСЃР°:"]
                        for pid, info in profiles.items():
                            status = "рџ”’" if info["is_blocked"] else ""
                            lines.append(f"вЂў {info['name']} {status}")
                        return True, "\n".join(lines)
                    return True, "РќРµС‚ Р·Р°РїРѕРјРЅРµРЅРЅС‹С… РіРѕР»РѕСЃРѕРІ."
        
        return False, ""
    
    def _process_loop(self):
        """РћР±СЂР°Р±РѕС‚РєР° Р°СѓРґРёРѕ РёР· РѕС‡РµСЂРµРґРё."""
        while self.is_running:
            try:
                item = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                audio, meta = item
            else:
                audio = item
                meta = {
                    "speaker_id": "unknown",
                    "speaker_name": "РќРµР·РЅР°РєРѕРјРµС†",
                    "priority": int(VoicePriority.GUEST) if SPEAKER_MANAGER_AVAILABLE else 2,
                    "confidence": 0.0,
                    "music_probability": 0.0,
                    "timestamp": time.time(),
                }

            kiwi_log("PROCESS", f"Got audio: {len(audio)/self.config.sample_rate:.1f}s, speaker={meta.get('speaker_id', 'unknown')}, music={meta.get('music_probability', 0.0):.2f}")
            
            kiwi_log("PROCESS", "Transcribing...")
            text = self._transcribe(audio)
            kiwi_log("PROCESS", f"Transcription result: {text}")
            
            if not text:
                continue
            
            # === РЎРўР РРњРРќР“: РџСЂРѕРІРµСЂСЏРµРј early wake detection ===
            if self._streaming_enabled and self._early_wake_detected:
                # РџСЂРѕРІРµСЂСЏРµРј, РЅРµ СѓСЃС‚Р°СЂРµР»Рѕ Р»Рё early detection
                time_since_early = time.time() - self._early_detected_at
                
                if time_since_early < self._streaming_early_timeout:
                    kiwi_log("STREAM", f"Using early detected wake word (age={time_since_early:.1f}s < {self._streaming_early_timeout}s)")
                    
                    # РћР±СЉРµРґРёРЅСЏРµРј early command СЃ С‚РµРєСѓС‰РµР№ С‚СЂР°РЅСЃРєСЂРёР±Р°С†РёРµР№
                    # Р”РµРґСѓРїР»РёРєР°С†РёСЏ: РµСЃР»Рё early command СѓР¶Рµ СЃРѕРґРµСЂР¶РёС‚СЃСЏ РІ final text вЂ” РЅРµ РґСѓР±Р»РёСЂРѕРІР°С‚СЊ
                    early_cmd = self._early_command.lower().strip()
                    final_text = text.lower().strip()
                    
                    # РџСЂРѕРІРµСЂСЏРµРј РїРµСЂРµСЃРµС‡РµРЅРёРµ вЂ” РёС‰РµРј early_cmd РІРЅСѓС‚СЂРё final_text
                    if early_cmd and early_cmd not in final_text:
                        # РЎРєР»РµРёРІР°РµРј: early + final
                        text = f"{self._early_command} {text}"
                        kiwi_log("STREAM", f"Combined: early='{self._early_command}' + final='{text[len(self._early_command)+1:]}'")
                    else:
                        kiwi_log("STREAM", "Early command already in final text, using final only")
                else:
                    kiwi_log("STREAM", f"Early detection expired (age={time_since_early:.1f}s >= {self._streaming_early_timeout}s), using final text only")
                
                # РЎР±СЂР°СЃС‹РІР°РµРј С„Р»Р°Рі Рё РѕС‡РёС‰Р°РµРј СЃС‚СЂРёРјРµСЂ
                self._reset_streaming_state()
            
            # РџСЂРѕРІРµСЂРєР° РЅР° С„Р°РЅС‚РѕРјРЅС‹Р№ С‚РµРєСЃС‚
            if self._is_phantom_text(text):
                continue
            
            kiwi_log("TEXT", f"Heard: {text}")

            # РћР±РЅРѕРІР»СЏРµРј СЃРЅРёРјРѕРє РїРѕСЃР»РµРґРЅРµРіРѕ РіРѕРІРѕСЂСЏС‰РµРіРѕ РґР»СЏ СЃРµСЂРІРёСЃР° (approval/policy)
            meta["text"] = text
            self._update_last_speaker_meta(meta, audio)

            # РћР±РЅРѕРІР»СЏРµРј voice context РґР»СЏ speaker manager, РµСЃР»Рё РµСЃС‚СЊ РІР°Р»РёРґРЅР°СЏ РёРґРµРЅС‚РёС„РёРєР°С†РёСЏ
            speaker_id = str(meta.get("speaker_id", "unknown"))
            speaker_name = str(meta.get("speaker_name", speaker_id))
            if speaker_id not in ("unknown", "self"):
                try:
                    self.update_voice_context(
                        speaker_id=speaker_id,
                        speaker_name=speaker_name,
                        priority=int(meta.get("priority", int(VoicePriority.GUEST) if SPEAKER_MANAGER_AVAILABLE else 2)),
                        confidence=float(meta.get("confidence", 0.0)),
                        command=text,
                    )
                except Exception as e:
                    kiwi_log("SPEAKER_MANAGER", f"Context update error: {e}", level="ERROR")
            
            if speaker_id != "self" and not self._is_owner_speaker(speaker_id):
                self._last_non_owner_activity_at = time.time()
            
            if self.on_speech:
                self.on_speech(text)
            
            # FIX: РџСЂРѕРІРµСЂСЏРµРј, Р±С‹Р»Р° Р»Рё СЂРµС‡СЊ РЅР°С‡Р°С‚Р° РІ СЂРµР¶РёРјРµ РґРёР°Р»РѕРіР° (РґР°Р¶Рµ РµСЃР»Рё РѕРЅ СѓР¶Рµ РёСЃС‚С‘Рє)
            # Р­С‚Рѕ РіР°СЂР°РЅС‚РёСЂСѓРµС‚, С‡С‚Рѕ СЂРµС‡СЊ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ РѕР±СЂР°Р±РѕС‚Р°РµС‚СЃСЏ РєР°Рє РєРѕРјР°РЅРґР°, РґР°Р¶Рµ РµСЃР»Рё
            # dialog_mode РёСЃС‚С‘Рє Рє РјРѕРјРµРЅС‚Сѓ С‚СЂР°РЅСЃРєСЂРёРїС†РёРё
            was_in_dialog = self._speech_started_in_dialog
            
            # РЎР±СЂР°СЃС‹РІР°РµРј С„Р»Р°Рі вЂ” РѕРЅ РёСЃРїРѕР»СЊР·РѕРІР°РЅ РґР»СЏ СЌС‚РѕР№ С„СЂР°Р·С‹
            self._speech_started_in_dialog = False
            
            # РџСЂРѕРІРµСЂСЏРµРј С‚РµРєСѓС‰РёР№ СЂРµР¶РёРј РґРёР°Р»РѕРіР°
            in_dialog = self._check_dialog_mode()
            
            # FIX: РћР±СЂР°Р±Р°С‚С‹РІР°РµРј РєР°Рє РєРѕРјР°РЅРґСѓ РµСЃР»Рё:
            # 1. РЎРµР№С‡Р°СЃ РІ СЂРµР¶РёРјРµ РґРёР°Р»РѕРіР° (in_dialog), РёР»Рё
            # 2. Р РµС‡СЊ Р±С‹Р»Р° РЅР°С‡Р°С‚Р° РІ СЂРµР¶РёРјРµ РґРёР°Р»РѕРіР° (was_in_dialog)
            if in_dialog or was_in_dialog:
                # Р’ СЂРµР¶РёРјРµ РґРёР°Р»РѕРіР° РёР»Рё СЂРµС‡СЊ РЅР°С‡Р°С‚Р° РІ dialog mode вЂ” РѕР±СЂР°Р±Р°С‚С‹РІР°РµРј РІСЃС‘ РєР°Рє РєРѕРјР°РЅРґС‹
                if was_in_dialog and not in_dialog:
                    kiwi_log("DIALOG", f"Processing as command (speech started in dialog mode): {text}")
                else:
                    kiwi_log("DIALOG", f"Processing: {text}")
                if self.on_wake_word:
                    self.on_wake_word(text)
            else:
                # РћР±С‹С‡РЅС‹Р№ СЂРµР¶РёРј вЂ” Р¶РґС‘Рј wake word
                allow_handsfree, reason = self._can_owner_skip_wake_word(meta)
                if allow_handsfree:
                    kiwi_log("HANDSFREE", f"Owner command without wake word: {text}")
                    self.activate_dialog_mode()
                    if self.on_wake_word:
                        self.on_wake_word(text)
                    continue
                
                is_address, command = self.detector.is_direct_address(text)
                
                if is_address:
                    # РђРєС‚РёРІРёСЂСѓРµРј СЂРµР¶РёРј РґРёР°Р»РѕРіР° РІ Р»СЋР±РѕРј СЃР»СѓС‡Р°Рµ (СЃ РєРѕРјР°РЅРґРѕР№ РёР»Рё Р±РµР·)
                    self.activate_dialog_mode()
                    
                    if command:
                        # Wake word + РєРѕРјР°РЅРґР° вЂ” СЃСЂР°Р·Сѓ РѕР±СЂР°Р±Р°С‚С‹РІР°РµРј
                        kiwi_log("KIWI", f"Wake word detected! Command: {command}")
                        if self.on_wake_word:
                            self.on_wake_word(command)
                    else:
                        # РўРѕР»СЊРєРѕ wake word Р±РµР· РєРѕРјР°РЅРґС‹ вЂ” Р¶РґС‘Рј РєРѕРјР°РЅРґСѓ РІ СЂРµР¶РёРјРµ РґРёР°Р»РѕРіР°
                        kiwi_log("KIWI", "Wake word detected! Waiting for command...")
                        # РќР• РІС‹Р·С‹РІР°РµРј callback вЂ” Р¶РґС‘Рј СЃР»РµРґСѓСЋС‰СѓСЋ С„СЂР°Р·Сѓ РІ СЂРµР¶РёРјРµ РґРёР°Р»РѕРіР°
                else:
                    # РќРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ wake-word + РІС‹СЃРѕРєР°СЏ РІРµСЂРѕСЏС‚РЅРѕСЃС‚СЊ РјСѓР·С‹РєРё + РЅРµРёР·РІРµСЃС‚РЅС‹Р№ РіРѕР»РѕСЃ
                    if reason in ("music", "guest_recent"):
                        kiwi_log("WAKE", f"Wake word required ({reason})")
                    if (
                        self._music_filter_enabled
                        and float(meta.get("music_probability", 0.0)) >= self._music_reject_threshold
                        and str(meta.get("speaker_id", "unknown")) == "unknown"
                    ):
                        kiwi_log("MUSIC", "Ignored transcription in idle mode (likely background music)")
    
    def _fix_transcription(self, text: str) -> str:
        """РђРІС‚РѕРёСЃРїСЂР°РІР»РµРЅРёРµ С‚СЂР°РЅСЃРєСЂРёРїС†РёРё: СЃР»РѕРІР°СЂСЊ Р·Р°РјРµРЅ + LLM."""
        if not text:
            return text
        
        original = text
        text_lower = text.lower()
        
        # 1. Р—Р°РјРµРЅР° РѕРїРµС‡Р°С‚РѕРє wake word (С‚РѕР»СЊРєРѕ С†РµР»С‹Рµ СЃР»РѕРІР°)
        for typo, correct in WAKE_WORD_TYPOS.items():
            # РСЃРїРѕР»СЊР·СѓРµРј word boundaries РґР»СЏ Р·Р°РјРµРЅС‹ С‚РѕР»СЊРєРѕ С†РµР»С‹С… СЃР»РѕРІ
            pattern = r'\b' + re.escape(typo) + r'\b'
            if re.search(pattern, text_lower):
                text_lower = re.sub(pattern, correct, text_lower)
                kiwi_log("FIX", f"Replaced '{typo}' -> '{correct}'")
        
        # Р’РѕСЃСЃС‚Р°РЅР°РІР»РёРІР°РµРј СЂРµРіРёСЃС‚СЂ РґР»СЏ wake word
        if "РєРёРІРё" in text_lower:
            text = text_lower
        
        # 2. Р”СЂСѓРіРёРµ С‡Р°СЃС‚С‹Рµ РёСЃРїСЂР°РІР»РµРЅРёСЏ
        text = text.replace("РєРёРµРІРµ", "РєРёРІРё")
        text = text.replace("РєРёРµРІРё", "РєРёРІРё")
        
        if text != original:
            kiwi_log("FIX", f"Dictionary correction: '{original}' -> '{text}'")
        
        # 3. LLM РёСЃРїСЂР°РІР»РµРЅРёРµ (РµСЃР»Рё РґРѕСЃС‚СѓРїРЅРѕ)
        if self._llm_fix_callback:
            try:
                llm_fixed = self._llm_fix_callback(text)
                if llm_fixed and llm_fixed != text:
                    kiwi_log("FIX", f"LLM correction: '{text}' -> '{llm_fixed}'")
                    text = llm_fixed
            except Exception as e:
                kiwi_log("FIX", f"LLM fix error: {e}", level="ERROR")
        
        return text
    
    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Р Р°СЃРїРѕР·РЅР°РµС‚ СЂРµС‡СЊ РёР· Р°СѓРґРёРѕ СЃ Р°РІС‚РѕРёСЃРїСЂР°РІР»РµРЅРёРµРј Рё С„РёР»СЊС‚СЂР°С†РёРµР№ РіР°Р»Р»СЋС†РёРЅР°С†РёР№."""
        try:
            duration = len(audio) / self.config.sample_rate
            kiwi_log("WHISPER", f"Input audio: shape={audio.shape}, duration={duration:.2f}s, range=[{audio.min():.3f}, {audio.max():.3f}]")
            
            # РЎР»РёС€РєРѕРј РєРѕСЂРѕС‚РєРѕРµ Р°СѓРґРёРѕ вЂ” С‡Р°СЃС‚Рѕ РјСѓСЃРѕСЂ
            if duration < 0.4:
                kiwi_log("WHISPER", f"Audio too short ({duration:.2f}s < 0.4s), skipping")
                return None
            
            segments, info = self.model.transcribe(
                audio,
                language="ru",
                task="transcribe",
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,  # Р’РђР–РќРћ: False С‡С‚РѕР±С‹ РЅРµ РіР°Р»Р»СЋС†РёРЅРёСЂРѕРІР°С‚СЊ РЅР° РѕСЃРЅРѕРІРµ РїСЂРµРґС‹РґСѓС‰РµРіРѕ С‚РµРєСЃС‚Р°
                initial_prompt=WHISPER_INITIAL_PROMPT,  # РџРѕРґСЃРєР°Р·РєР° РґР»СЏ Р»СѓС‡С€РµРіРѕ СЂР°СЃРїРѕР·РЅР°РІР°РЅРёСЏ
                no_speech_threshold=0.85,  # РџРѕСЂРѕРі РІРµСЂРѕСЏС‚РЅРѕСЃС‚Рё "РЅРµС‚ СЂРµС‡Рё"
            )
            
            kiwi_log("WHISPER", f"Detected language: {info.language}, probability: {info.language_probability:.2f}")

            text_parts = []
            for segment in segments:
                # === Р¤РР›Р¬РўР РђР¦РРЇ Р“РђР›Р›Р®Р¦РРќРђР¦РР™ ===
                no_speech = getattr(segment, 'no_speech_prob', 0.0)
                avg_logprob = getattr(segment, 'avg_logprob', 0.0)
                
                kiwi_log("WHISPER", f"Segment: [{segment.start:.2f}s -> {segment.end:.2f}s] '{segment.text}' (no_speech={no_speech:.2f}, avg_logprob={avg_logprob:.2f})")

                if no_speech > 0.85:
                    kiwi_log("WHISPER", f"Segment skipped (no_speech): {no_speech:.2f} > 0.6", level="WARNING")
                    continue
                
                # РџСЂРѕРїСѓСЃРєР°РµРј СЃРµРіРјРµРЅС‚С‹ СЃ РѕС‡РµРЅСЊ РЅРёР·РєРѕР№ СѓРІРµСЂРµРЅРЅРѕСЃС‚СЊСЋ (РіР°Р»Р»СЋС†РёРЅР°С†РёРё)
                if avg_logprob < -1.0:
                    kiwi_log("WHISPER", f"Segment skipped: avg_logprob={avg_logprob:.2f} < -1.0 (likely hallucination)", level="WARNING")
                    continue
                
                # Segment claims longer than actual audio = hallucination
                if segment.end > duration * 2.0 + 1.0:
                    kiwi_log("WHISPER", f"Segment skipped: end={segment.end:.2f}s >> audio={duration:.2f}s (timestamp hallucination)", level="WARNING")
                    continue
                
                text_parts.append(segment.text)
            
            full_text = " ".join(text_parts).strip()
            
            # Check against known Whisper hallucination patterns
            if full_text:
                text_lower = full_text.strip().lower()
                for pattern in WHISPER_HALLUCINATION_PATTERNS:
                    if text_lower.startswith(pattern):
                        kiwi_log("WHISPER", f"Hallucination filtered: '{full_text}' (matched: '{pattern}')", level="WARNING")
                        return None
            
            if full_text:
                # РђРІС‚РѕРёСЃРїСЂР°РІР»РµРЅРёРµ С‚СЂР°РЅСЃРєСЂРёРїС†РёРё
                full_text = self._fix_transcription(full_text)
            
            return full_text if full_text else None
            
        except Exception as e:
            kiwi_log("LISTENER", f"Transcription error: {e}", level="ERROR")
            import traceback
            traceback.print_exc()
            return None


def main():
    """РўРµСЃС‚РѕРІС‹Р№ Р·Р°РїСѓСЃРє СЃР»СѓС€Р°С‚РµР»СЏ."""
    
    def on_wake_word(command: str):
        print(f"\n[KIWI] KIWI ACTIVATED! Command: {command}\n")
    
    def on_speech(text: str):
        pass
    
    listener = KiwiListener(
        config=ListenerConfig(),
        on_wake_word=on_wake_word,
        on_speech=on_speech,
    )
    
    try:
        listener.start()
        print("\nSay 'Kiwi, ...' to activate. Press Ctrl+C to exit.\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[BYE] Stopping...")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()

