import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import threading
from Model.inference_module import start_inference_loop
from alert_system.notification import start_alert_checker
from node.node_wifi import run_listener_for_all

import os
import time
import sys
import torch
import sqlite3

import asyncio
import json
import queue
try:
    import websockets
except ImportError:
    websockets = None

from Model.base_model_panns import (
    PANNsCNN10,
    TransferClassifier,
    infer_audio,
    get_device, 
    get_label_dict
)


device = get_device()
label_dict = get_label_dict(root_dir='./Dataset/Dataset')
real_time_folder = "./Input_data/real_input"

panns_model = PANNsCNN10('./Model/pretrained/Cnn10.pth').to(device)
classifier_model = TransferClassifier(input_dim=512, num_classes=len(label_dict))
classifier_model.load_state_dict(torch.load('Model/classifier_model.pth', map_location=device))
classifier_model.to(device)
classifier_model.eval()

# === WebSocket broadcast setup ===
# Outbox for cross-thread handoff from DB poller → WS event loop
WS_OUTBOX: "queue.Queue[str]" = queue.Queue()
# Will hold the asyncio loop used by the WS server thread
_WS_LOOP: asyncio.AbstractEventLoop | None = None
# Connected clients set (lives in the WS loop)
_WS_CLIENTS: set = set()

async def _broadcast_loop():
    """Runs inside WS event loop; pulls JSON strings from WS_OUTBOX and sends to all clients."""
    while True:
        # Block in a thread-friendly way: use run_in_executor to avoid blocking the loop
        payload = await asyncio.get_running_loop().run_in_executor(None, WS_OUTBOX.get)
        # Ship to connected clients
        stale = []
        if _WS_CLIENTS:
            for ws in list(_WS_CLIENTS):
                try:
                    await ws.send(payload)
                except Exception:
                    stale.append(ws)
        # Cleanup stale clients
        for ws in stale:
            try:
                _WS_CLIENTS.discard(ws)
                await ws.close()
            except Exception:
                pass

async def _ws_handler(websocket, path):
    """Accepts client connections and keeps them registered for broadcast."""
    _WS_CLIENTS.add(websocket)
    try:
        # Optional: initial hello
        await websocket.send(json.dumps({"type": "hello", "msg": "connected", "clients": len(_WS_CLIENTS)}))
        # Keep the connection open; consume pings/keepalive
        async for _ in websocket:
            # We don't expect client messages; ignore or implement commands later
            pass
    finally:
        _WS_CLIENTS.discard(websocket)

def start_ws_server(host: str = "0.0.0.0", port: int = 8765):
    """Starts a WebSocket server and a broadcast task in its own asyncio loop (runs in a thread)."""
    global _WS_LOOP
    if websockets is None:
        logging.error("The 'websockets' package is not installed. Install with: pip install websockets")
        return
    _WS_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(_WS_LOOP)
    server_coro = websockets.serve(_ws_handler, host, port, ping_interval=20, ping_timeout=20)
    server = _WS_LOOP.run_until_complete(server_coro)
    logging.info(f"[WSServer] WebSocket server started on ws://{host}:{port}")
    # Background broadcaster
    _WS_LOOP.create_task(_broadcast_loop())
    try:
        _WS_LOOP.run_forever()
    finally:
        server.close()
        _WS_LOOP.run_until_complete(server.wait_closed())
        _WS_LOOP.close()


# === Runtime configuration (BLE input + WebSocket) ===
# SQLite alerts DB path (existing project convention: DB/alerts.db)
DB_PATH = os.environ.get("ALERTS_DB_PATH", "DB/alerts.db")

def start_ws_db_publisher(poll_interval: float = 1.0):
    """
    Polls the alerts table and enqueues new rows to WS_OUTBOX for WebSocket broadcast.
    Assumes an 'alerts' table with columns:
      rowid (implicit), room_id, category, decibel, priority, created_at
    """
    logging.info(f"[WSPublisher] Start polling DB={DB_PATH} for new alerts")
    while not os.path.exists(DB_PATH):
        logging.info("[WSPublisher] Waiting for DB to be created...")
        time.sleep(1.5)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Initialize last seen rowid
    cur.execute("SELECT IFNULL(MAX(rowid), 0) as max_id FROM alerts")
    last_id = cur.fetchone()["max_id"] or 0
    logging.info(f"[WSPublisher] Initial last_id={last_id}")

    try:
        while True:
            cur.execute("""
                SELECT rowid, room_id, category, decibel, priority, created_at
                FROM alerts
                WHERE rowid > ?
                ORDER BY rowid ASC
            """, (last_id,))
            rows = cur.fetchall()

            if rows:
                for r in rows:
                    payload = {
                        "type": "alert",
                        "id": int(r["rowid"]),
                        "room_id": r["room_id"],
                        "category": r["category"],
                        "decibel": r["decibel"],
                        "priority": r["priority"],
                        "created_at": r["created_at"]
                    }
                    try:
                        WS_OUTBOX.put_nowait(json.dumps(payload))
                        last_id = int(r["rowid"])
                        logging.info(f"[WSPublisher] Enqueued alert #{last_id} for WS broadcast")
                    except Exception as e:
                        logging.warning(f"[WSPublisher] Failed to enqueue alert: {e}")

            time.sleep(poll_interval)
    finally:
        conn.close()


#  threading을 사용하면 병렬로 구현이 가능함. 
#  node simulator, start_inference_loop, start_alert_checker 가 동시에 돌아간다 

if __name__ == '__main__':
    ws_host = os.environ.get("WS_HOST", "0.0.0.0")
    try:
        ws_port = int(os.environ.get("WS_PORT", "8765"))
    except ValueError:
        ws_port = 8765
    threading.Thread(
        target=start_ws_server,
        kwargs={"host": ws_host, "port": ws_port},
        name="WSServerThread",
        daemon=True
    ).start()

    threading.Thread(
        target=start_inference_loop,
        args=(real_time_folder, panns_model, classifier_model, label_dict, device),
        name="InferenceThread"
    ).start()

    # [DISABLED] Node simulator is disabled for production with ESP32 BLE input.
    # threading.Thread(
    #     target=start_node_simulation,
    #     name="NodeSimThread"
    # ).start()

    # Run Flask HTTP receiver in the main thread (blocking)
    run_listener_for_all(real_time_folder=real_time_folder, host="0.0.0.0", port=5050)

    threading.Thread(
        target=start_alert_checker,
        name="AlertThread"
    ).start()

    threading.Thread(
        target=start_ws_db_publisher,
        name="WSPublisherThread",
        daemon=True
    ).start()

    # [DISABLED] Real-time visualization will be handled by the Android app; keep main thread free.
    # start_db_visualization()