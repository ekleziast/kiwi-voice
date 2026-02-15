#!/usr/bin/env python3
"""
Speaker Identification Module for Kiwi Voice

Phase 1: Detect self (TTS echo) to prevent Kiwi from responding to herself
Phase 2: Identify known speakers (owner vs guests)
"""

import os
import json
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", module="pyannote")
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from kiwi.utils import kiwi_log

# pyannote.audio для speaker embedding (опционально, fallback есть)
try:
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    kiwi_log("SPEAKER-ID", "pyannote.audio not available, using fallback method", level="WARNING")


@dataclass
class SpeakerProfile:
    """Профиль говорящего."""
    name: str
    embeddings: List[List[float]]  # Список векторов (для усреднения)
    priority: str  # "owner", "guest", "self" (Kiwi TTS), "unknown"
    created_at: str = ""
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Возвращает усреднённый embedding."""
        if not self.embeddings:
            return None
        return np.mean(np.array(self.embeddings), axis=0)


class SpeakerIdentifier:
    """
    Идентификатор говорящих на основе voice embeddings.
    
    Использует pyannote/embedding для извлечения признаков голоса.
    Fallback: простое сравнение спектральных характеристик если pyannote недоступен.
    """
    
    # Пороги сходства (косинусное расстояние)
    SIMILARITY_THRESHOLD = 0.70  # Выше = тот же говорящий
    SELF_SIMILARITY_THRESHOLD = 0.75  # Строже для собственного голоса
    
    def __init__(self, profiles_dir: Optional[str] = None):
        """
        Args:
            profiles_dir: Директория для хранения профилей (по умолчанию: папка модуля)
        """
        if profiles_dir is None:
            from kiwi import PROJECT_ROOT
            profiles_dir = os.path.join(PROJECT_ROOT, "voice_profiles")
        
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiles: Dict[str, SpeakerProfile] = {}
        self.embedding_model = None
        self._model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ленивая загрузка: модель загрузится при первом вызове extract_embedding()
        kiwi_log("SPEAKER-ID", f"Initialized (lazy loading enabled, device={self.device})")
        
        # Загружаем существующие профили
        self._load_profiles()
        
        # Если нет профиля self (Kiwi TTS) - создадим при первой генерации
        if "self" not in self.profiles:
            kiwi_log("SPEAKER-ID", "No self-profile found. Will create on first TTS.")
    
    def _get_profile_path(self) -> Path:
        """Путь к файлу профилей."""
        return self.profiles_dir / "profiles.json"
    
    def _load_profiles(self):
        """Загружает профили из JSON."""
        profile_path = self._get_profile_path()
        if not profile_path.exists():
            return
        
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for profile_id, profile_data in data.get("profiles", {}).items():
                self.profiles[profile_id] = SpeakerProfile(**profile_data)
            
            kiwi_log("SPEAKER-ID", f"Loaded {len(self.profiles)} profiles")
        except Exception as e:
            kiwi_log("SPEAKER-ID", f"Error loading profiles: {e}", level="ERROR")
    
    def _save_profiles(self):
        """Сохраняет профили в JSON."""
        try:
            data = {
                "profiles": {
                    pid: asdict(p) for pid, p in self.profiles.items()
                }
            }
            with open(self._get_profile_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            kiwi_log("SPEAKER-ID", f"Error saving profiles: {e}", level="ERROR")
    
    def _ensure_model_loaded(self):
        """Загружает модель при первом вызове (ленивая загрузка)."""
        if self._model_loaded:
            return
        
        if not PYANNOTE_AVAILABLE:
            kiwi_log("SPEAKER-ID", "pyannote not available, using fallback", level="WARNING")
            self._model_loaded = True
            return
        
        try:
            kiwi_log("SPEAKER-ID", "Loading embedding model (first use)...")
            from kiwi import PROJECT_ROOT
            local_model_path = Path(PROJECT_ROOT) / "models" / "pyannote-embedding"
            if local_model_path.exists():
                self.embedding_model = PretrainedSpeakerEmbedding(
                    str(local_model_path),
                    device=self.device
                )
                kiwi_log("SPEAKER-ID", f"Loaded embedding model from local path on {self.device}")
            else:
                kiwi_log("SPEAKER-ID", f"Local model not found at {local_model_path}, trying HuggingFace...", level="WARNING")
                self.embedding_model = PretrainedSpeakerEmbedding(
                    "pyannote/embedding",
                    device=self.device
                )
                kiwi_log("SPEAKER-ID", f"Loaded embedding model from HuggingFace on {self.device}")
            self._model_loaded = True
        except Exception as e:
            kiwi_log("SPEAKER-ID", f"Failed to load embedding model: {e}", level="ERROR")
            self.embedding_model = None
            self._model_loaded = True
    
    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Извлекает embedding из аудио.
        
        Args:
            audio: numpy array с аудио (float32, [-1, 1])
            sample_rate: частота дискретизации
            
        Returns:
            embedding вектор или None если не удалось
        """
        # Ленивая загрузка модели при первом вызове
        self._ensure_model_loaded()
        
        if self.embedding_model is None:
            return self._extract_fallback_embedding(audio, sample_rate)
        
        try:
            # pyannote ожидает torch tensor shape: (channels, samples)
            if len(audio.shape) == 1:
                waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            else:
                waveform = torch.tensor(audio, dtype=torch.float32)
            
            # Проверяем длину (нужно минимум ~1 секунды)
            if waveform.shape[1] < sample_rate * 0.5:
                # Слишком короткое аудио
                return None
            
            with torch.no_grad():
                embedding = self.embedding_model(waveform)
            
            # Проверяем тип результата (torch tensor или numpy array)
            if hasattr(embedding, 'cpu'):
                return embedding.cpu().numpy().flatten()
            else:
                return np.array(embedding).flatten()
            
        except Exception as e:
            kiwi_log("SPEAKER-ID", f"Embedding extraction error: {e}", level="ERROR")
            return self._extract_fallback_embedding(audio, sample_rate)
    
    def _extract_fallback_embedding(self, audio: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """
        Fallback метод если pyannote недоступен.
        Использует простые спектральные признаки.
        """
        try:
            # Простая спектральная характеристика
            from numpy.fft import rfft
            
            # Разбиваем на окна
            window_size = int(sample_rate * 0.025)  # 25ms
            hop_size = int(sample_rate * 0.010)     # 10ms
            
            # Извлекаем MFCC-подобные признаки (упрощённые)
            frames = []
            for i in range(0, len(audio) - window_size, hop_size):
                frame = audio[i:i + window_size]
                # Оконная функция
                frame = frame * np.hanning(len(frame))
                # FFT
                spectrum = np.abs(rfft(frame))
                # Логарифмическая шкала
                log_spectrum = np.log(spectrum + 1e-10)
                frames.append(log_spectrum[:40])  # Берём первые 40 бинов
            
            if not frames:
                return None
            
            # Усредняем по времени
            mean_spectrum = np.mean(frames, axis=0)
            # Нормализуем
            mean_spectrum = (mean_spectrum - np.mean(mean_spectrum)) / (np.std(mean_spectrum) + 1e-10)
            
            return mean_spectrum
            
        except Exception as e:
            kiwi_log("SPEAKER-ID", f"Fallback embedding error: {e}", level="ERROR")
            return None
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Вычисляет косинусное сходство между двумя векторами."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def identify_speaker(
        self, 
        audio: np.ndarray, 
        sample_rate: int = 16000,
        exclude_self: bool = False
    ) -> Tuple[str, float]:
        """
        Определяет говорящего по аудио.
        
        Args:
            audio: numpy array с аудио
            sample_rate: частота дискретизации
            exclude_self: если True, не возвращает "self" (для проверки эха)
            
        Returns:
            (speaker_id, confidence)
            speaker_id: "self", "owner", "guest", "unknown"
        """
        embedding = self.extract_embedding(audio, sample_rate)
        
        if embedding is None:
            return "unknown", 0.0
        
        # Если нет профилей - возвращаем unknown
        if not self.profiles:
            return "unknown", 0.0
        
        best_match = None
        best_score = -1.0
        
        for profile_id, profile in self.profiles.items():
            if exclude_self and profile_id == "self":
                continue
            
            profile_embedding = profile.get_average_embedding()
            if profile_embedding is None:
                continue
            
            similarity = self.cosine_similarity(embedding, profile_embedding)
            
            if similarity > best_score:
                best_score = similarity
                best_match = profile_id
        
        # Определяем порог в зависимости от типа
        if best_match == "self":
            threshold = self.SELF_SIMILARITY_THRESHOLD
        else:
            threshold = self.SIMILARITY_THRESHOLD
        
        if best_score >= threshold and best_match:
            return best_match, best_score
        
        return "unknown", best_score
    
    def is_self_speaking(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, float]:
        """
        Проверяет, говорит ли сама Киви (эхо от TTS).
        
        Returns:
            (is_self, confidence)
        """
        if "self" not in self.profiles:
            # Нет профиля - считаем что не эхо
            return False, 0.0
        
        speaker_id, confidence = self.identify_speaker(audio, sample_rate)
        
        if speaker_id == "self" and confidence >= self.SELF_SIMILARITY_THRESHOLD:
            return True, confidence
        
        return False, confidence
    
    def add_profile_sample(
        self, 
        profile_id: str, 
        audio: np.ndarray, 
        sample_rate: int = 16000,
        name: Optional[str] = None,
        priority: str = "guest"
    ) -> bool:
        """
        Добавляет образец голоса к профилю.
        
        Args:
            profile_id: ID профиля ("self", "owner", "guest_1", etc.)
            audio: аудио семпл
            sample_rate: частота дискретизации
            name: Человекочитаемое имя
            priority: "owner", "guest", "self"
            
        Returns:
            True если успешно
        """
        embedding = self.extract_embedding(audio, sample_rate)
        
        if embedding is None:
            kiwi_log("SPEAKER-ID", f"Failed to extract embedding for {profile_id}", level="ERROR")
            return False
        
        if profile_id not in self.profiles:
            from datetime import datetime
            self.profiles[profile_id] = SpeakerProfile(
                name=name or profile_id,
                embeddings=[],
                priority=priority,
                created_at=datetime.now().isoformat()
            )
        
        self.profiles[profile_id].embeddings.append(embedding.tolist())
        
        # Ограничиваем количество семплов (берём последние 5)
        if len(self.profiles[profile_id].embeddings) > 5:
            self.profiles[profile_id].embeddings = self.profiles[profile_id].embeddings[-5:]
        
        self._save_profiles()
        kiwi_log("SPEAKER-ID", f"Added sample to '{profile_id}', total: {len(self.profiles[profile_id].embeddings)}")
        
        return True
    
    def create_self_profile(self, tts_audio: np.ndarray, sample_rate: int = 24000) -> bool:
        """
        Создаёт профиль собственного голоса (TTS) для фильтрации эха.
        
        Args:
            tts_audio: аудио сгенерированное TTS
            sample_rate: частота дискретизации
            
        Returns:
            True если успешно
        """
        # Конвертируем sample_rate если нужно
        if sample_rate != 16000:
            import librosa
            tts_audio = librosa.resample(
                tts_audio.astype(np.float32), 
                orig_sr=sample_rate, 
                target_sr=16000
            )
        
        success = self.add_profile_sample(
            profile_id="self",
            audio=tts_audio,
            sample_rate=16000,
            name="Kiwi TTS",
            priority="self"
        )
        
        if success:
            kiwi_log("SPEAKER-ID", "Created self-profile from TTS audio")
        
        return success
    
    def calibrate_self_from_tts(self, tts_text: str = "Привет, я Киви. Это тестовая фраза для калибровки."):
        """
        Генерирует тестовую фразу через TTS и создаёт профиль.
        Используется для начальной калибровки.
        
        Returns:
            True если успешно
        """
        try:
            # Импортируем здесь чтобы избежать циклических зависимостей
            from kiwi.tts.piper import PiperTTS
            
            tts = PiperTTS()
            audio, sr = tts.synthesize(tts_text)
            
            if audio is not None:
                return self.create_self_profile(audio, sr)
            
        except Exception as e:
            kiwi_log("SPEAKER-ID", f"TTS calibration error: {e}", level="ERROR")
        
        return False
    
    def get_profile_info(self) -> Dict:
        """Возвращает информацию о загруженных профилях."""
        return {
            pid: {
                "name": p.name,
                "priority": p.priority,
                "samples": len(p.embeddings)
            }
            for pid, p in self.profiles.items()
        }


# Тестирование модуля
if __name__ == "__main__":
    print("[TEST] Speaker Identifier Test")
    
    sid = SpeakerIdentifier()
    
    # Если нет self-профиля - создаём
    if "self" not in sid.profiles:
        print("[TEST] Creating self profile...")
        sid.calibrate_self_from_tts()
    
    print(f"[TEST] Profiles: {sid.get_profile_info()}")
