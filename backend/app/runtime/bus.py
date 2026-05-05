from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, DefaultDict
from collections import defaultdict


@dataclass
class BusMessage:
    topic: str
    payload: Any


class MessageBus:
    """A minimal in-process async pub/sub bus.

    This is intentionally simple:
    - topics are strings
    - subscribers receive BusMessage objects via an asyncio.Queue
    """

    def __init__(self) -> None:
        self._subs: DefaultDict[str, List[asyncio.Queue]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subs[topic].append(q)
        return q

    async def unsubscribe(self, topic: str, q: asyncio.Queue) -> None:
        async with self._lock:
            if topic in self._subs and q in self._subs[topic]:
                self._subs[topic].remove(q)

    async def publish(self, topic: str, payload: Any) -> None:
        async with self._lock:
            queues = list(self._subs.get(topic, []))
        msg = BusMessage(topic=topic, payload=payload)
        for q in queues:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # If a subscriber is too slow, we drop instead of blocking the whole system.
                pass

    async def broadcast(self, payload: Any) -> None:
        # Broadcast is a convention topic.
        await self.publish("broadcast", payload)
