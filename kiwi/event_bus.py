"""
Event-Based Architecture - Центральная шина событий

Заменяет polling на event-driven подход:
- Компоненты публикуют события вместо прямых вызовов
- Обработчики подписываются на интересующие события
- Асинхронная обработка через очереди
"""

import threading
import queue
import time
import uuid
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import traceback

from kiwi.utils import kiwi_log


class EventType(Enum):
    """Типы событий системы."""
    # Аудио события
    SPEECH_STARTED = auto()
    SPEECH_ENDED = auto()
    WAKE_WORD_DETECTED = auto()
    COMMAND_RECEIVED = auto()
    
    # Состояние
    STATE_CHANGED = auto()
    DIALOG_MODE_ENTERED = auto()
    DIALOG_MODE_EXITED = auto()
    
    # OpenClaw
    LLM_THINKING_STARTED = auto()
    LLM_THINKING_ENDED = auto()
    LLM_TOKEN_RECEIVED = auto()
    LLM_RESPONSE_COMPLETE = auto()
    
    # TTS
    TTS_STARTED = auto()
    TTS_CHUNK_GENERATED = auto()
    TTS_CHUNK_PLAYING = auto()
    TTS_ENDED = auto()
    TTS_BARGE_IN = auto()
    
    # VAD
    VAD_SPEECH_DETECTED = auto()
    VAD_SILENCE_DETECTED = auto()
    
    # AEC
    AEC_ECHO_DETECTED = auto()
    AEC_ECHO_CANCELLED = auto()
    
    # Speaker
    SPEAKER_IDENTIFIED = auto()
    SPEAKER_BLOCKED = auto()
    
    # Ошибки
    ERROR_TRANSCRIPTION = auto()
    ERROR_TTS = auto()
    ERROR_LLM = auto()
    
    # Системные
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    CONFIG_RELOADED = auto()


@dataclass
class Event:
    """Событие в системе."""
    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = "unknown"
    
    def get(self, key: str, default=None):
        """Получить значение из payload."""
        return self.payload.get(key, default)
    
    def __repr__(self):
        return f"Event({self.type.name}, id={self.event_id}, src={self.source})"


class EventHandler:
    """Обёртка для обработчика событий с приоритетом и фильтрами."""
    
    def __init__(
        self,
        callback: Callable[[Event], None],
        priority: int = 0,
        filter_func: Optional[Callable[[Event], bool]] = None,
        async_mode: bool = True
    ):
        self.callback = callback
        self.priority = priority
        self.filter_func = filter_func
        self.async_mode = async_mode
        self.call_count = 0
        self.total_time = 0.0
    
    def can_handle(self, event: Event) -> bool:
        """Проверяет, может ли обработчик обработать событие."""
        if self.filter_func is None:
            return True
        try:
            return self.filter_func(event)
        except Exception as e:
            kiwi_log("EVENT", f"Filter error: {e}", level="ERROR")
            return False
    
    def handle(self, event: Event):
        """Вызывает обработчик."""
        start = time.time()
        try:
            self.callback(event)
        except Exception as e:
            kiwi_log("EVENT", f"Handler error: {e}", level="ERROR")
            traceback.print_exc()
        finally:
            self.call_count += 1
            self.total_time += time.time() - start
    
    @property
    def avg_time(self) -> float:
        """Среднее время выполнения."""
        return self.total_time / max(1, self.call_count)


class EventBus:
    """
    Центральная шина событий.
    
    Паттерн Pub/Sub для loose coupling компонентов.
    """
    
    def __init__(self, max_queue_size: int = 1000, num_workers: int = 2):
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._event_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker_threads: List[threading.Thread] = []
        self._num_workers = num_workers
        self._running = False
        self._stop_event = threading.Event()
        
        # Статистика
        self._events_published = 0
        self._events_processed = 0
        self._events_dropped = 0
        self._event_times: Dict[EventType, List[float]] = defaultdict(list)
        
        # История (для отладки)
        self._history: List[Event] = []
        self._max_history = 100
        
        self._lock = threading.RLock()
    
    def start(self):
        """Запускает обработку событий."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        for i in range(self._num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"EventWorker-{i}",
                daemon=True
            )
            worker.start()
            self._worker_threads.append(worker)
        
        kiwi_log("EVENT_BUS", f"Started with {self._num_workers} workers")
    
    def stop(self):
        """Останавливает обработку событий."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        # Добавляем пустые события чтобы разблокировать worker'ы
        for _ in range(self._num_workers):
            try:
                self._event_queue.put(None, block=False)
            except queue.Full:
                pass
        
        for worker in self._worker_threads:
            worker.join(timeout=2.0)
        
        self._worker_threads.clear()
        kiwi_log("EVENT_BUS", "Stopped")
    
    def _worker_loop(self):
        """Цикл обработки событий worker'ом."""
        while not self._stop_event.is_set():
            try:
                event = self._event_queue.get(timeout=0.1)
                if event is None:  # Сигнал остановки
                    break
                
                self._process_event(event)
                
            except queue.Empty:
                continue
            except Exception as e:
                kiwi_log("EVENT_BUS", f"Worker error: {e}", level="ERROR")
    
    def _process_event(self, event: Event):
        """Обрабатывает одно событие."""
        start_time = time.time()
        
        with self._lock:
            handlers = list(self._handlers.get(event.type, []))
        
        # Сортируем по приоритету (выше = раньше)
        handlers.sort(key=lambda h: h.priority, reverse=True)
        
        for handler in handlers:
            if not handler.can_handle(event):
                continue
            
            if handler.async_mode:
                # Асинхронная обработка в отдельном потоке
                threading.Thread(
                    target=handler.handle,
                    args=(event,),
                    daemon=True
                ).start()
            else:
                # Синхронная обработка
                handler.handle(event)
        
        # Статистика
        processing_time = time.time() - start_time
        with self._lock:
            self._events_processed += 1
            self._event_times[event.type].append(processing_time)
            if len(self._event_times[event.type]) > 100:
                self._event_times[event.type] = self._event_times[event.type][-50:]
            
            # История
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
    
    def publish(
        self,
        event_type: EventType,
        payload: Optional[Dict[str, Any]] = None,
        source: str = "unknown",
        wait: bool = False
    ) -> Optional[Event]:
        """
        Публикует событие в шину.
        
        Args:
            event_type: Тип события
            payload: Данные события
            source: Источник события
            wait: Если True, ждёт обработки (для критических событий)
        
        Returns:
            Созданное событие (если wait=True) или None
        """
        event = Event(
            type=event_type,
            payload=payload or {},
            source=source
        )
        
        try:
            if wait:
                # Синхронная публикация
                self._process_event(event)
                return event
            else:
                # Асинхронная публикация
                self._event_queue.put(event, block=False)
                with self._lock:
                    self._events_published += 1
                return event
                
        except queue.Full:
            with self._lock:
                self._events_dropped += 1
            kiwi_log("EVENT_BUS", f"Queue full, event dropped: {event_type.name}", level="WARNING")
            return None
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
        priority: int = 0,
        filter_func: Optional[Callable[[Event], bool]] = None,
        async_mode: bool = True
    ) -> str:
        """
        Подписывает обработчик на событие.
        
        Args:
            event_type: Тип события
            callback: Функция-обработчик
            priority: Приоритет (выше = раньше вызывается)
            filter_func: Фильтр событий
            async_mode: Асинхронная обработка
        
        Returns:
            ID подписки (для отписки)
        """
        handler = EventHandler(callback, priority, filter_func, async_mode)
        
        with self._lock:
            self._handlers[event_type].append(handler)
        
        handler_id = f"{event_type.name}_{id(handler)}"
        kiwi_log("EVENT_BUS", f"Subscribed {callback.__name__} to {event_type.name} (priority={priority})")
        
        return handler_id
    
    def subscribe_multi(
        self,
        event_types: List[EventType],
        callback: Callable[[Event], None],
        priority: int = 0,
        async_mode: bool = True
    ) -> List[str]:
        """Подписывает один обработчик на несколько событий."""
        ids = []
        for event_type in event_types:
            handler_id = self.subscribe(event_type, callback, priority, async_mode=async_mode)
            ids.append(handler_id)
        return ids
    
    def unsubscribe(self, event_type: EventType, handler_id: str) -> bool:
        """Отписывает обработчик."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            for handler in handlers:
                if f"{event_type.name}_{id(handler)}" == handler_id:
                    handlers.remove(handler)
                    return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику шины."""
        with self._lock:
            avg_times = {
                event_type.name: sum(times) / len(times)
                for event_type, times in self._event_times.items()
                if times
            }
            
            handler_counts = {
                event_type.name: len(handlers)
                for event_type, handlers in self._handlers.items()
            }
            
            return {
                'events_published': self._events_published,
                'events_processed': self._events_processed,
                'events_dropped': self._events_dropped,
                'queue_size': self._event_queue.qsize(),
                'handlers_count': handler_counts,
                'avg_processing_time': avg_times,
                'drop_rate': self._events_dropped / max(1, self._events_published),
            }
    
    def get_recent_events(self, count: int = 10) -> List[Event]:
        """Возвращает последние события."""
        with self._lock:
            return self._history[-count:]
    
    def clear_history(self):
        """Очищает историю событий."""
        with self._lock:
            self._history.clear()


# Глобальный экземпляр шины событий
_event_bus_instance: Optional[EventBus] = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    """Возвращает глобальный экземпляр шины событий (singleton)."""
    global _event_bus_instance
    
    if _event_bus_instance is None:
        with _event_bus_lock:
            if _event_bus_instance is None:
                _event_bus_instance = EventBus()
    
    return _event_bus_instance


def publish(
    event_type: EventType,
    payload: Optional[Dict[str, Any]] = None,
    source: str = "unknown",
    wait: bool = False
) -> Optional[Event]:
    """Удобная функция для публикации событий."""
    return get_event_bus().publish(event_type, payload, source, wait)


def subscribe(
    event_type: EventType,
    callback: Callable[[Event], None],
    priority: int = 0,
    async_mode: bool = True
) -> str:
    """Удобная функция для подписки на события."""
    return get_event_bus().subscribe(event_type, callback, priority, async_mode=async_mode)


# Примеры использования
def example_usage():
    """Пример использования шины событий."""
    bus = EventBus()
    bus.start()
    
    # Подписываемся на события
    def on_speech_started(event: Event):
        print(f"Speech started at {event.timestamp}")
    
    def on_command(event: Event):
        command = event.get('command', 'unknown')
        print(f"Command received: {command}")
    
    bus.subscribe(EventType.SPEECH_STARTED, on_speech_started, priority=10)
    bus.subscribe(EventType.COMMAND_RECEIVED, on_command)
    
    # Публикуем события
    bus.publish(EventType.SPEECH_STARTED, {'volume': 0.5}, source="listener")
    bus.publish(EventType.COMMAND_RECEIVED, {'command': 'привет'}, source="listener")
    
    # Статистика
    print(f"\nStats: {bus.get_stats()}")
    
    time.sleep(1)
    bus.stop()


if __name__ == "__main__":
    example_usage()
