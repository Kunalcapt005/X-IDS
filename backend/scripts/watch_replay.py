from __future__ import annotations

import argparse
import asyncio
import json

import websockets


async def watch(session_id: int, host: str, port: int) -> None:
    url = f"ws://{host}:{port}/sessions/{session_id}/stream"
    print(f"Connecting to {url}")
    async with websockets.connect(url, ping_interval=20, ping_timeout=20) as websocket:
        async for message in websocket:
            payload = json.loads(message)
            event_type = payload.get("type")
            if event_type == "alert":
                print(
                    f"ALERT #{payload.get('alert_id')}: "
                    f"{payload.get('predicted_label')} | "
                    f"{payload.get('severity')} | "
                    f"{float(payload.get('confidence', 0)):.1%}"
                )
            else:
                print(json.dumps(payload, indent=2))
            if event_type in {"session_completed", "session_failed", "session_cancelled"}:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch an X-IDS replay WebSocket")
    parser.add_argument("session_id", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    asyncio.run(watch(args.session_id, args.host, args.port))
