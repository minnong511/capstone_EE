import asyncio
import json
import sqlite3
import time
from typing import Set, Optional
import logging

import websockets
from websockets.server import WebSocketServerProtocol


"""
•	WebSocket 서버 기동 (start_websocket_server)
•	다른 스레드에서 안전하게 전송 (broadcast_json)
•	DB를 tail 해서 개별 알림 이벤트를 순차 전송 (start_db_event_publisher)
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# --- Global state for cross-thread access ---
CONNECTED: Set[WebSocketServerProtocol] = set()
WS_LOOP: Optional[asyncio.AbstractEventLoop] = None

async def _handler(ws: WebSocketServerProtocol):
    """Accept connections and keep them until closed. Push-only server."""
    CONNECTED.add(ws)
    try:
        await ws.send(json.dumps({"type": "hello", "message": "connected"}))
        async for _ in ws:
            # Ignore client messages for now.
            pass
    finally:
        CONNECTED.discard(ws)


async def _broadcast_text(text: str):
    """Coroutine that sends `text` to all connected clients."""
    if not CONNECTED:
        return
    tasks = []
    for ws in list(CONNECTED):
        if ws.closed:
            CONNECTED.discard(ws)
            continue
        tasks.append(ws.send(text))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

# --- SQLite helper ---
def _connect_db():
    """
    Create a SQLite connection and set safe PRAGMAs for concurrent access.
    """
    conn = sqlite3.connect('./DB/alerts.db', timeout=2.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=2000;")
    except Exception:
        pass
    return conn

def broadcast_json(payload: dict):
    """
    Thread-safe entrypoint to broadcast a JSON object to all connected clients.
    Safe to call from non-async threads (e.g., inference loop thread).
    Silently no-ops if the server isn't up yet.
    """
    global WS_LOOP
    if WS_LOOP is None or not WS_LOOP.is_running():
        return
    text = json.dumps(payload, ensure_ascii=False)
    fut = asyncio.run_coroutine_threadsafe(_broadcast_text(text), WS_LOOP)
    # Avoid raising exceptions in caller thread
    def _swallow(f):
        try:
            f.result()
        except Exception:
            pass
    fut.add_done_callback(_swallow)
    logging.info(f"[WebSocket] Sent payload: {payload}")

async def start_websocket_server(host: str = "0.0.0.0", port: int = 8765):
    """
    Start the WebSocket server with the `websockets` package and run forever.
    Stores the running loop globally so other threads can call broadcast_json().
    """
    global WS_LOOP
    WS_LOOP = asyncio.get_running_loop()
    print(f"[WebSocket] Server started at ws://{host}:{port}")
    async def heartbeat():
        while True:
            num_clients = len(CONNECTED)
            logging.info(f"[WebSocket] 서버 작동 중 ... (연결된 클라이언트: {num_clients}명)")
            await asyncio.sleep(5)
    asyncio.create_task(heartbeat())
    async with websockets.serve(
        _handler, host, port, ping_interval=20, ping_timeout=20, max_queue=32
    ):
        await asyncio.Future()  # run forever


# --- DB event publisher ---
def start_db_event_publisher(poll_ms: int = 500, send_history: bool = False, batch_limit: int = 200):
    """
    Tail the `alerts` table and broadcast each new row as an individual JSON event via WebSocket.
    Designed to run in a background thread (daemon).

    Args:
        poll_ms: polling interval in milliseconds.
        send_history: if True, send existing rows from startup; if False, start from current MAX(id).
        batch_limit: maximum rows to send per iteration (protects from huge bursts).

    Payload schema:
    {
      "type": "alert_event",
      "id": 123,
      "room_id": "Room102",
      "mic_id": "ESP32_01",
      "category": "doorbell",
      "decibel": 71,
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "file": "ESP32_01_Room102_20250908-113210-012345.wav"
    }
    """
    interval_s = max(0.05, poll_ms / 1000.0)

    # Establish starting checkpoint
    conn = _connect_db()
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(id), 0) FROM alerts")
    max_id = cur.fetchone()[0] or 0
    conn.close()
    last_id = 0 if send_history else max_id

    while True:
        try:
            conn = _connect_db()
            cur = conn.cursor()
            cur.execute(f"""
                SELECT id, room_id, mic_id, category, decibel, created_at, file
                FROM alerts
                WHERE id > ?
                ORDER BY id ASC
                LIMIT {int(batch_limit)}
            """, (last_id,))
            rows = cur.fetchall()
            conn.close()

            # Broadcast each new row
            for (rid, room_id, mic_id, category, decibel, created_at, file_) in rows:
                payload = {
                    "type": "alert_event",
                    "id": int(rid),
                    "room_id": room_id,
                    "mic_id": mic_id,
                    "category": category,
                    "decibel": int(decibel) if decibel is not None else None,
                    "timestamp": created_at,
                    "file": file_
                }
                broadcast_json(payload)
                last_id = rid
        except Exception:
            # Swallow exceptions to keep the publisher alive; real logs can be added by caller
            pass
        finally:
            time.sleep(interval_s)