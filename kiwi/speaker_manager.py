#!/usr/bin/env python3
"""
Speaker Manager - Voice Priority + Access Control

Расширенная система управления голосами:
- Иерархия приоритетов (OWNER > FRIEND > GUEST > BLOCKED)
- Hot cache для быстрой идентификации
- Auto-learning новых голосов
- Контекст последнего говорящего
"""

import os
import json
import time
import threading
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import IntEnum
from datetime import datetime, timedelta

import numpy as np

from kiwi.utils import kiwi_log

try:
    from kiwi.speaker_id import SpeakerIdentifier, SpeakerProfile
    BASE_AVAILABLE = True
except ImportError:
    BASE_AVAILABLE = False
    kiwi_log("SPEAKER_MANAGER", "speaker_id module not available", level="WARNING")


class VoicePriority(IntEnum):
    """Приоритеты голосов (меньше = выше приоритет)."""
    SELF = -1      # Kiwi TTS (только для фильтрации эха)
    OWNER = 0      # Владелец - максимальный приоритет
    FRIEND = 1     # Запомненные друзья/знакомые
    GUEST = 2      # Неизвестные гости
    BLOCKED = 99   # Чёрный список (полный игнор)


@dataclass
class VoiceContext:
    """Контекст последнего говорящего."""
    speaker_id: str = ""
    speaker_name: str = ""
    priority: VoicePriority = VoicePriority.GUEST
    confidence: float = 0.0
    last_command: str = ""
    timestamp: float = 0.0
    is_processing: bool = False
    
    def is_valid(self) -> bool:
        """Контекст валиден в течение 30 секунд."""
        return (
            self.speaker_id and 
            time.time() - self.timestamp < 30.0
        )
    
    def update(self, speaker_id: str, speaker_name: str, priority: VoicePriority, confidence: float, command: str = ""):
        """Обновляет контекст."""
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.priority = priority
        self.confidence = confidence
        self.last_command = command
        self.timestamp = time.time()
    
    def clear(self):
        """Очищает контекст."""
        self.speaker_id = ""
        self.speaker_name = ""
        self.priority = VoicePriority.GUEST
        self.confidence = 0.0
        self.last_command = ""
        self.timestamp = 0.0


@dataclass
class ExtendedSpeakerProfile:
    """Расширенный профиль говорящего."""
    # Из speaker_id
    name: str
    embeddings: List[List[float]]
    priority: str  # "owner", "guest", "self"
    created_at: str = ""
    
    # Новые поля
    speaker_id: str = ""  # Уникальный ID (owner, friend_имя, guest_UUID)
    display_name: str = ""  # Отображаемое имя
    is_blocked: bool = False
    auto_learned: bool = False
    last_seen: str = ""
    confidence_threshold: float = 0.70
    
    def get_base_profile(self):
        """Конвертирует в базовый SpeakerProfile."""
        if not BASE_AVAILABLE:
            return None
        return SpeakerProfile(
            name=self.name,
            embeddings=self.embeddings,
            priority=self.priority,
            created_at=self.created_at
        )
    
    @classmethod
    def from_base(cls, base, speaker_id: str = "") -> "ExtendedSpeakerProfile":
        """Создаёт из базового профиля."""
        return cls(
            name=base.name,
            embeddings=base.embeddings,
            priority=base.priority,
            created_at=base.created_at,
            speaker_id=speaker_id or base.name.lower().replace(" ", "_"),
            display_name=base.name,
            last_seen=datetime.now().isoformat()
        )


class SpeakerManager:
    """
    Менеджер управления голосами с приоритетами.
    
    Особенности:
    - Hot cache в RAM для <10ms идентификации
    - Auto-learning при высокой уверенности (>0.85)
    - Контекст последнего говорящего (30 сек)
    - OWNER команды прерывают текущие задачи
    """
    
    # Настройки
    AUTO_LEARN_THRESHOLD = 0.85  # Авто-запоминание при такой уверенности
    IDENTIFY_THRESHOLD = 0.70     # Минимум для распознавания
    HOT_CACHE_SIZE = 10           # Размер горячего кэша
    CONTEXT_TIMEOUT = 30.0        # Таймаут контекста (сек)
    
    # OWNER ID (можно настроить)
    OWNER_ID = "owner"
    OWNER_NAME = "Owner"

    def __init__(self, profiles_dir: Optional[str] = None, base_identifier: Optional["SpeakerIdentifier"] = None, owner_name: Optional[str] = None):
        """
        Args:
            profiles_dir: Директория для профилей
        """
        if owner_name:
            self.OWNER_NAME = owner_name
        self.profiles_dir = Path(profiles_dir) if profiles_dir else Path("voice_profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Базовая система идентификации
        # Use shared SpeakerIdentifier if provided, otherwise create new
        self.base_identifier: Optional[SpeakerIdentifier] = base_identifier
        if self.base_identifier is None and BASE_AVAILABLE:
            try:
                self.base_identifier = SpeakerIdentifier(str(self.profiles_dir))
                kiwi_log("SPEAKER_MANAGER", "Base identifier loaded (new instance)")
            except Exception as e:
                kiwi_log("SPEAKER_MANAGER", f"Failed to load base identifier: {e}", level="ERROR")
        elif self.base_identifier is not None:
            kiwi_log("SPEAKER_MANAGER", "Using shared base identifier")
        
        # Расширенные профили
        self.profiles: Dict[str, ExtendedSpeakerProfile] = {}
        self._load_extended_profiles()
        
        # Hot cache для быстрой идентификации
        self._hot_cache: Dict[str, np.ndarray] = {}
        self._hot_cache_lock = threading.Lock()
        
        # Контекст последнего говорящего
        self.voice_context = VoiceContext()
        
        # Временный кэш для обучения
        self._temp_cache: Dict[str, List[np.ndarray]] = {}  # speaker_id -> embeddings
        self._temp_cache_lock = threading.Lock()
        
        kiwi_log("SPEAKER_MANAGER", f"Initialized with {len(self.profiles)} profiles")
    
    def _get_profiles_path(self) -> Path:
        """Путь к файлу расширенных профилей."""
        return self.profiles_dir / "extended_profiles.json"
    
    def _load_extended_profiles(self):
        """Загружает расширенные профили."""
        path = self._get_profiles_path()
        if not path.exists():
            return
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for pid, profile_data in data.get("profiles", {}).items():
                self.profiles[pid] = ExtendedSpeakerProfile(**profile_data)
            
            kiwi_log("SPEAKER_MANAGER", f"Loaded {len(self.profiles)} extended profiles")
        except Exception as e:
            kiwi_log("SPEAKER_MANAGER", f"Error loading profiles: {e}", level="ERROR")
    
    def _save_extended_profiles(self):
        """Сохраняет расширенные профили."""
        try:
            data = {
                "profiles": {
                    pid: asdict(p) for pid, p in self.profiles.items()
                }
            }
            with open(self._get_profiles_path(), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            kiwi_log("SPEAKER_MANAGER", f"Error saving profiles: {e}", level="ERROR")
    
    def _generate_speaker_id(self, name: str) -> str:
        """Генерирует уникальный speaker_id из имени."""
        # Очищаем имя
        clean = "".join(c for c in name.lower() if c.isalnum() or c == "_")
        return f"friend_{clean}" if clean else f"guest_{int(time.time())}"
    
    def register_owner(self, audio: np.ndarray, sample_rate: int = 16000, name: str = None) -> bool:
        """
        Регистрирует OWNER.
        
        Args:
            audio: Аудио семпл
            sample_rate: Частота дискретизации
            name: Имя (по умолчанию OWNER_NAME)
            
        Returns:
            True если успешно
        """
        name = name or self.OWNER_NAME
        speaker_id = self.OWNER_ID
        
        kiwi_log("SPEAKER_MANAGER", f"Registering OWNER: {name} ({speaker_id})")
        
        if self.base_identifier:
            success = self.base_identifier.add_profile_sample(
                profile_id=speaker_id,
                audio=audio,
                sample_rate=sample_rate,
                name=name,
                priority="owner"
            )
            
            if success:
                # Создаём/обновляем расширенный профиль
                self.profiles[speaker_id] = ExtendedSpeakerProfile(
                    name=name,
                    embeddings=self.base_identifier.profiles[speaker_id].embeddings if speaker_id in self.base_identifier.profiles else [],
                    priority="owner",
                    speaker_id=speaker_id,
                    display_name=name,
                    is_blocked=False,
                    last_seen=datetime.now().isoformat()
                )
                self._save_extended_profiles()
                
                # Добавляем в hot cache
                embedding = self.base_identifier.extract_embedding(audio, sample_rate)
                if embedding is not None:
                    with self._hot_cache_lock:
                        self._hot_cache[speaker_id] = embedding
                
                kiwi_log("SPEAKER_MANAGER", "OWNER registered successfully")
                return True
        
        return False
    
    def add_friend(self, audio: np.ndarray, sample_rate: int = 16000, name: str = None, 
                   auto_learn: bool = False) -> Tuple[bool, str]:
        """
        Добавляет друга/знакомого.
        
        Args:
            audio: Аудио семпл
            sample_rate: Частота дискретизации
            name: Имя человека
            auto_learn: Авто-запоминание без явного добавления
            
        Returns:
            (success, speaker_id)
        """
        if not name:
            return False, "Name required"
        
        speaker_id = self._generate_speaker_id(name)
        
        kiwi_log("SPEAKER_MANAGER", f"Adding FRIEND: {name} ({speaker_id})")
        
        if self.base_identifier:
            success = self.base_identifier.add_profile_sample(
                profile_id=speaker_id,
                audio=audio,
                sample_rate=sample_rate,
                name=name,
                priority="guest"
            )
            
            if success:
                self.profiles[speaker_id] = ExtendedSpeakerProfile(
                    name=name,
                    embeddings=self.base_identifier.profiles[speaker_id].embeddings if speaker_id in self.base_identifier.profiles else [],
                    priority="guest",
                    speaker_id=speaker_id,
                    display_name=name,
                    is_blocked=False,
                    auto_learned=auto_learn,
                    last_seen=datetime.now().isoformat()
                )
                self._save_extended_profiles()
                
                # Hot cache
                embedding = self.base_identifier.extract_embedding(audio, sample_rate)
                if embedding is not None:
                    with self._hot_cache_lock:
                        self._hot_cache[speaker_id] = embedding
                
                kiwi_log("SPEAKER_MANAGER", f"FRIEND added: {name}")
                return True, speaker_id
        
        return False, "Failed to add"
    
    def block_speaker(self, speaker_id: str) -> bool:
        """
        Блокирует голос (добавляет в чёрный список).
        
        Args:
            speaker_id: ID голоса для блокировки
            
        Returns:
            True если найден и заблокирован
        """
        # OWNER не блокируется
        if speaker_id == self.OWNER_ID:
            kiwi_log("SPEAKER_MANAGER", "Cannot block OWNER", level="WARNING")
            return False
        
        if speaker_id in self.profiles:
            self.profiles[speaker_id].is_blocked = True
            self._save_extended_profiles()
            
            # Удаляем из hot cache
            with self._hot_cache_lock:
                self._hot_cache.pop(speaker_id, None)
            
            kiwi_log("SPEAKER_MANAGER", f"BLOCKED: {speaker_id}")
            return True
        
        # Если не найден в профилях - создаём временную запись
        if speaker_id:
            self.profiles[speaker_id] = ExtendedSpeakerProfile(
                name="Blocked",
                embeddings=[],
                priority="guest",
                speaker_id=speaker_id,
                display_name="Заблокированный",
                is_blocked=True,
                last_seen=datetime.now().isoformat()
            )
            self._save_extended_profiles()
            kiwi_log("SPEAKER_MANAGER", f"BLOCKED (temp): {speaker_id}")
            return True
        
        return False
    
    def unblock_speaker(self, speaker_id: str) -> bool:
        """
        Разблокирует голос.
        
        Args:
            speaker_id: ID голоса
            
        Returns:
            True если найден и разблокирован
        """
        if speaker_id in self.profiles:
            self.profiles[speaker_id].is_blocked = False
            self._save_extended_profiles()
            kiwi_log("SPEAKER_MANAGER", f"UNBLOCKED: {speaker_id}")
            return True
        return False
    
    def identify_speaker_fast(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, VoicePriority, float, str]:
        """
        Быстрая идентификация говорящего с приоритетом.
        
        Использует hot cache для мгновенного ответа.
        
        Returns:
            (speaker_id, priority, confidence, display_name)
        """
        # Проверяем hot cache
        if self.base_identifier:
            embedding = self.base_identifier.extract_embedding(audio, sample_rate)
            
            if embedding is not None:
                # Сначала проверяем hot cache
                with self._hot_cache_lock:
                    cache_items = list(self._hot_cache.items())
                
                best_id = None
                best_score = 0.0
                
                for sid, cached_emb in cache_items:
                    if sid in self.profiles and self.profiles[sid].is_blocked:
                        continue  # Пропускаем заблокированных
                    
                    score = self.base_identifier.cosine_similarity(embedding, cached_emb)
                    if score > best_score:
                        best_score = score
                        best_id = sid
                
                # Проверяем порог для hot cache
                if best_id and best_score >= self.IDENTIFY_THRESHOLD:
                    priority = self._get_priority(best_id)
                    name = self.profiles.get(best_id, ExtendedSpeakerProfile(name=best_id, embeddings=[], priority="guest")).display_name or best_id
                    return best_id, priority, best_score, name
                
                # Если не найден в кэше - полное сканирование
                speaker_id, score = self.base_identifier.identify_speaker(audio, sample_rate)
                
                if speaker_id != "unknown" and score >= self.IDENTIFY_THRESHOLD:
                    priority = self._get_priority(speaker_id)
                    
                    # Проверяем блокировку
                    if speaker_id in self.profiles and self.profiles[speaker_id].is_blocked:
                        return speaker_id, VoicePriority.BLOCKED, score, self.profiles[speaker_id].display_name
                    
                    # Добавляем в hot cache
                    with self._hot_cache_lock:
                        self._hot_cache[speaker_id] = embedding
                        # Ограничиваем размер кэша
                        if len(self._hot_cache) > self.HOT_CACHE_SIZE:
                            oldest = next(iter(self._hot_cache))
                            del self._hot_cache[oldest]
                    
                    name = self.profiles.get(speaker_id, ExtendedSpeakerProfile(name=speaker_id, embeddings=[], priority="guest")).display_name or speaker_id
                    return speaker_id, priority, score, name
        
        # Неизвестный голос
        return "unknown", VoicePriority.GUEST, 0.0, "Незнакомец"
    
    def _get_priority(self, speaker_id: str) -> VoicePriority:
        """Определяет приоритет по speaker_id."""
        if speaker_id == self.OWNER_ID:
            return VoicePriority.OWNER
        if speaker_id == "self":
            return VoicePriority.SELF
        if speaker_id in self.profiles and self.profiles[speaker_id].is_blocked:
            return VoicePriority.BLOCKED
        if speaker_id in self.profiles and self.profiles[speaker_id].priority == "guest":
            return VoicePriority.FRIEND
        if speaker_id.startswith("friend_"):
            return VoicePriority.FRIEND
        if speaker_id.startswith("guest_"):
            return VoicePriority.GUEST
        if speaker_id == "unknown":
            return VoicePriority.GUEST
        return VoicePriority.GUEST
    
    def auto_learn_voice(self, audio: np.ndarray, speaker_id: str, sample_rate: int = 16000) -> bool:
        """
        Автоматически обучается на новый голос при высокой уверенности.
        
        Args:
            audio: Аудио
            speaker_id: Текущий ID (может быть "unknown")
            sample_rate: Частота
            
        Returns:
            True если добавлен новый профиль
        """
        # Проверяем порог
        current_id, confidence = speaker_id, 0.0
        
        if self.base_identifier:
            _, confidence = self.base_identifier.identify_speaker(audio, sample_rate)
        
        if confidence < self.AUTO_LEARN_THRESHOLD:
            return False
        
        # Добавляем в временный кэш
        with self._temp_cache_lock:
            if speaker_id not in self._temp_cache:
                self._temp_cache[speaker_id] = []
            
            if self.base_identifier:
                embedding = self.base_identifier.extract_embedding(audio, sample_rate)
                if embedding is not None:
                    self._temp_cache[speaker_id].append(embedding)
            
            # Если накоплено достаточно семплов (3-5) - сохраняем
            if len(self._temp_cache[speaker_id]) >= 3:
                # Генерируем имя на основе ID
                name = f"Guest_{speaker_id[-4:]}" if speaker_id.startswith("guest_") else speaker_id
                
                # Добавляем как временного друга
                if self.base_identifier:
                    self.base_identifier.add_profile_sample(
                        profile_id=speaker_id,
                        audio=audio,
                        sample_rate=sample_rate,
                        name=name,
                        priority="guest"
                    )
                
                self.profiles[speaker_id] = ExtendedSpeakerProfile(
                    name=name,
                    embeddings=self.base_identifier.profiles[speaker_id].embeddings if (self.base_identifier and speaker_id in self.base_identifier.profiles) else [],
                    priority="guest",
                    speaker_id=speaker_id,
                    display_name=name,
                    is_blocked=False,
                    auto_learned=True,
                    last_seen=datetime.now().isoformat()
                )
                self._save_extended_profiles()
                
                # Очищаем кэш
                del self._temp_cache[speaker_id]
                
                kiwi_log("SPEAKER_MANAGER", f"AUTO-LEARNED: {name} ({speaker_id})")
                return True
        
        return False
    
    def update_context(self, speaker_id: str, speaker_name: str, priority: VoicePriority, confidence: float, command: str = ""):
        """Обновляет контекст последнего говорящего."""
        self.voice_context.update(speaker_id, speaker_name, priority, confidence, command)
        kiwi_log("SPEAKER_MANAGER", f"Context updated: {speaker_name} ({priority.name}), confidence={confidence:.2f}")
    
    def get_context_speaker_id(self) -> Optional[str]:
        """Возвращает ID последнего говорящего (если контекст валиден)."""
        if self.voice_context.is_valid():
            return self.voice_context.speaker_id
        return None
    
    def is_owner(self, speaker_id: str) -> bool:
        """Проверяет, является ли это OWNER."""
        return speaker_id == self.OWNER_ID
    
    def is_blocked(self, speaker_id: str) -> bool:
        """Проверяет, заблокирован ли голос."""
        if speaker_id == self.OWNER_ID:
            return False
        if speaker_id in self.profiles:
            return self.profiles[speaker_id].is_blocked
        return False
    
    def can_execute_command(self, speaker_id: str) -> Tuple[bool, str]:
        """
        Проверяет, может ли голос выполнять команды.
        
        Returns:
            (allowed, reason)
        """
        if speaker_id == "unknown":
            return False, "Неизвестный голос"
        
        if self.is_blocked(speaker_id):
            return False, "Голос заблокирован"
        
        priority = self._get_priority(speaker_id)
        
        if priority == VoicePriority.BLOCKED:
            return False, "Голос в чёрном списке"
        
        if priority == VoicePriority.SELF:
            return False, "Это Kiwi (эхо)"
        
        return True, "Разрешено"
    
    def get_profile_info(self) -> Dict:
        """Возвращает информацию о профилях."""
        return {
            pid: {
                "name": p.display_name,
                "priority": p.priority,
                "is_blocked": p.is_blocked,
                "auto_learned": p.auto_learned,
                "samples": len(p.embeddings),
                "last_seen": p.last_seen
            }
            for pid, p in self.profiles.items()
        }
    
    def who_am_i(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Отвечает 'Кто это говорит?'."""
        speaker_id, priority, confidence, name = self.identify_speaker_fast(audio, sample_rate)
        
        if speaker_id == self.OWNER_ID:
            return f"Это ты, {self.OWNER_NAME}! ({confidence:.0%} уверенность)"
        elif speaker_id == "self":
            return "Это я, Киви!"
        elif priority == VoicePriority.BLOCKED:
            return "Это заблокированный голос"
        elif speaker_id.startswith("friend_"):
            return f"Это {name} ({confidence:.0%} уверенность)"
        elif speaker_id != "unknown":
            return f"Это {name} ({confidence:.0%} уверенность)"
        else:
            return "Я не узнала этот голос"


# Тестирование
if __name__ == "__main__":
    print("[TEST] Speaker Manager Test")
    
    manager = SpeakerManager()
    
    print(f"[TEST] Profiles: {len(manager.profiles)}")
    print(f"[TEST] Profile info: {manager.get_profile_info()}")
    
    # Проверка OWNER защиты
    print(f"[TEST] Owner blocked: {manager.is_blocked('owner')}")
    print(f"[TEST] Can execute (owner): {manager.can_execute_command('owner')}")
