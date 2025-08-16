# save as ble_receive_wav.py
import asyncio
import json
import os
import sys
from datetime import datetime
from bleak import BleakScanner, BleakClient

# Windows 환경 BLE 이벤트 루프 설정
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

DEVICE_NAME = "Sensor-01"
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_FILE_META_UUID = "12345678-1234-5678-1234-56789abcdef2"
CHAR_FILE_DATA_UUID = "12345678-1234-5678-1234-56789abcdef3"

SAVE_DIR = r"C:\github_project_vscode\capstone_EE\capstone_EE\BLE_Test\ble_save_test"

class FileReceiver:
    def __init__(self):
        self.expected_size = None
        self.sensor = None
        self.buffer = bytearray()
        self.last_seq = -1
        self.ready = False
        self.current_id = None  # 현재 파일 ID 저장

    def handle_meta(self, data: bytes):
        meta = json.loads(data.decode("utf-8"))
        fid = meta.get("id", None)

        # 같은 ID의 META가 오면 무시
        if fid is not None and fid == self.current_id:
            print(f"[PC] Duplicate META for id={fid}, ignoring.")
            return

        self.current_id = fid
        self.sensor = meta.get("sensor", "Sensor")
        self.expected_size = int(meta.get("size", 0))
        self.buffer = bytearray()
        self.last_seq = -1
        self.ready = True
        print(f"[PC] META received: sensor={self.sensor}, size={self.expected_size} bytes, id={self.current_id}")

    def handle_data(self, data: bytes):
        if not self.ready:
            print("[PC] WARN: data arrived before META, ignoring chunk")
            return False

        if len(data) < 3:
            return False

        seq = data[0] | (data[1] << 8)
        flags = data[2]
        payload = data[3:]

        if seq != self.last_seq + 1:
            # 첫 chunk 누락은 WARN 대신 INFO로 완화
            if self.last_seq == -1:
                print(f"[PC] INFO: first chunk seq={seq} (expected 0)")
            else:
                print(f"[PC] WARN: unexpected seq {seq} (last {self.last_seq})")

        self.last_seq = seq
        self.buffer.extend(payload)

        if flags & 0x01:  # EOF
            print(f"[PC] EOF received. Total bytes={len(self.buffer)}")
            return True
        return False

    def save_file(self):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{self.sensor}_{ts}.wav"
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, filename)
        with open(path, "wb") as f:
            f.write(self.buffer)
        return path

    def reset_for_next(self):
        # 다음 파일 받을 준비
        self.expected_size = None
        self.sensor = None
        self.buffer = bytearray()
        self.last_seq = -1
        self.ready = False
        self.current_id = None

async def main():
    print("[PC] Scanning for device...")
    dev = None
    devices = await BleakScanner.discover(timeout=5.0)
    for d in devices:
        if d.name == DEVICE_NAME:
            dev = d
            break
    if not dev:
        print(f"[PC] Device '{DEVICE_NAME}' not found.")
        return

    receiver = FileReceiver()

    async with BleakClient(dev) as client:
        loop = asyncio.get_running_loop()

        def meta_cb(_, data: bytearray):
            receiver.handle_meta(bytes(data))

        def data_cb_sender(_, data: bytearray):
            eof = receiver.handle_data(bytes(data))
            if eof:
                total = len(receiver.buffer)
                if receiver.expected_size and total != receiver.expected_size:
                    print(f"[PC] WARN: size mismatch (expected {receiver.expected_size}, got {total})")

                path = receiver.save_file()
                print(f"[PC] Saved: {path}")

                async def delete_later(p):
                    await asyncio.sleep(10)
                    try:
                        os.remove(p)
                        print(f"[PC] Deleted after 10s: {p}")
                    except Exception as e:
                        print(f"[PC] Delete error: {e}")

                loop.create_task(delete_later(path))
                receiver.reset_for_next()

        await client.start_notify(CHAR_FILE_META_UUID, meta_cb)
        await client.start_notify(CHAR_FILE_DATA_UUID, data_cb_sender)

        print("[PC] Subscribed. Waiting for files... (Ctrl+C to exit)")
        try:
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            print("\n[PC] Stopped by user.")

        await client.stop_notify(CHAR_FILE_DATA_UUID)
        await client.stop_notify(CHAR_FILE_META_UUID)

if __name__ == "__main__":
    asyncio.run(main())
