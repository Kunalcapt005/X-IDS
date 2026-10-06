from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import WebSocket


class StreamHub:
    """In-memory WebSocket fan-out for the single-process development server."""

    def __init__(self) -> None:
        self._clients: dict[int, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def register(self, session_id: int, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients[session_id].add(websocket)

    async def unregister(self, session_id: int, websocket: WebSocket) -> None:
        async with self._lock:
            clients = self._clients.get(session_id)
            if clients is None:
                return
            clients.discard(websocket)
            if not clients:
                self._clients.pop(session_id, None)

    async def publish(self, session_id: int, payload: dict) -> None:
        async with self._lock:
            clients = list(self._clients.get(session_id, set()))

        stale: list[WebSocket] = []
        for websocket in clients:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)

        if stale:
            async with self._lock:
                clients = self._clients.get(session_id, set())
                for websocket in stale:
                    clients.discard(websocket)
                if not clients:
                    self._clients.pop(session_id, None)


stream_hub = StreamHub()
