# BLE-only multi-device WAV receiver
# - ESP32 쪽에서 임계치 감지 + 2초 녹음 + WAV 패키징 + BLE 전송을 수행
# - PC는 BLE 연결/유지와 파일 저장만 담당
#
# 기능:
# 1) 여러 ESP32(예: Sensor-01, Sensor-02, Sensor-03)를 동시 접속 유지
# 2) 각 기기에서 META(JSON) -> DATA(청크) -> EOF 수신 시 WAV 저장
# 3) 저장 경로: ./Input_data/real_input (상대 경로)
#
# 종료: Ctrl+C

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

from bleak import BleakScanner, BleakClient, BLEDevice

# -------------------------
# 연결/프로토콜 설정
# -------------------------
DEVICE_NAMES: List[str] = ["Sensor-01", "Sensor-02", "Sensor-03"]  # 필요 수만큼 이름 추가/수정
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_FILE_META_UUID = "12345678-1234-5678-1234-56789abcdef2"
CHAR_FILE_DATA_UUID = "12345678-1234-5678-1234-56789abcdef3"

# -------------------------
# 상대 경로 저장 디렉터리
# -------------------------
ROOT_DIR = Path(__file__).resolve().parent
TARGET_DIR = ROOT_DIR / "Input_data" / "real_input"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 파일 수신 상태기
# -------------------------
class FileReceiver:
    def __init__(self, sensor_name: str):
        self.sensor_name = sensor_name
        self.expected_size: Optional[int] = None
        self.current_id: Optional[int] = None
        self.buffer = bytearray()
        self.last_seq = -1
        self.ready = False

    def handle_meta(self, data: bytes):
        try:
            meta = json.loads(data.decode("utf-8"))
        except Exception as e:
            print(f"[{self.sensor_name}] META decode error: {e}")
            return

        fid = meta.get("id", None)
        if fid is not None and fid == self.current_id:
            print(f"[{self.sensor_name}] Duplicate META for id={fid}, ignoring.")
            return

        self.current_id = fid
        self.expected_size = int(meta.get("size", 0))
        self.buffer = bytearray()
        self.last_seq = -1
        self.ready = True
        print(f"[{self.sensor_name}] META: size={self.expected_size} bytes, id={self.current_id}")

    def handle_data(self, data: bytes) -> bool:
        if not self.ready:
            print(f"[{self.sensor_name}] WARN: data before META, ignoring chunk")
            return False

        if len(data) < 3:
            return False

        seq = data[0] | (data[1] << 8)
        flags = data[2]
        payload = data[3:]

        if seq != self.last_seq + 1:
            if self.last_seq == -1:
                print(f"[{self.sensor_name}] INFO: first chunk seq={seq} (expected 0)")
            else:
                print(f"[{self.sensor_name}] WARN: unexpected seq {seq} (last {self.last_seq})")

        self.last_seq = seq
        self.buffer.extend(payload)

        if flags & 0x01:  # EOF
            print(f"[{self.sensor_name}] EOF: total bytes={len(self.buffer)}")
            return True
        return False

    def save_and_reset(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{self.sensor_name}_{ts}.wav"
        path = TARGET_DIR / filename
        with open(path, "wb") as f:
            f.write(self.buffer)

        # size check
        if self.expected_size and len(self.buffer) != self.expected_size:
            print(f"[{self.sensor_name}] WARN: size mismatch (expected {self.expected_size}, got {len(self.buffer)})")

        # reset
        self.expected_size = None
        self.current_id = None
        self.buffer = bytearray()
        self.last_seq = -1
        self.ready = False

        print(f"[{self.sensor_name}] Saved: {path}")
        return path

# -------------------------
# 디바이스 연결/유지 루프
# -------------------------
async def device_worker(device_name: str):
    """
    주어진 이름의 디바이스를 주기적으로 검색 -> 연결 -> notify 구독 -> 수신/저장.
    연결이 끊기면 자동으로 재검색/재연결을 시도.
    """
    receiver = FileReceiver(sensor_name=device_name)

    while True:
        try:
            print(f"[{device_name}] Scanning for device...")
            dev: Optional[BLEDevice] = None
            devices = await BleakScanner.discover(timeout=5.0)
            for d in devices:
                if d.name == device_name:
                    dev = d
                    break

            if not dev:
                print(f"[{device_name}] Not found. Retry in 3s...")
                await asyncio.sleep(3.0)
                continue

            print(f"[{device_name}] Found {dev.address}. Connecting...")
            async with BleakClient(dev) as client:
                print(f"[{device_name}] Connected.")

                def meta_cb(_, data: bytearray):
                    receiver.handle_meta(bytes(data))

                def data_cb(_, data: bytearray):
                    eof = receiver.handle_data(bytes(data))
                    if eof:
                        receiver.save_and_reset()

                await client.start_notify(CHAR_FILE_META_UUID, meta_cb)
                await client.start_notify(CHAR_FILE_DATA_UUID, data_cb)

                print(f"[{device_name}] Subscribed. Waiting for files...")
                try:
                    while True:
                        await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    print(f"[{device_name}] Cancelled.")
                    break
                except KeyboardInterrupt:
                    print(f"[{device_name}] KeyboardInterrupt.")
                    break
                finally:
                    try:
                        await client.stop_notify(CHAR_FILE_DATA_UUID)
                        await client.stop_notify(CHAR_FILE_META_UUID)
                    except Exception:
                        pass

        except Exception as e:
            print(f"[{device_name}] Error: {e}. Reconnecting in 3s...")
            await asyncio.sleep(3.0)
            continue

# -------------------------
# 엔트리포인트
# -------------------------
async def main():
    # Windows 이벤트 루프 정책 (Windows일 때만)
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    print("[PC] Target save dir:", TARGET_DIR)
    tasks = [asyncio.create_task(device_worker(name)) for name in DEVICE_NAMES]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n[PC] Stopping...")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
