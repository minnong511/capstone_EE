"""
===============================================================================
BLE WAV Receiver (PC-side)
===============================================================================

[코드 목차]
  0) 개요와 입출력 구조(이 파일의 역할)
  1) 기본 상수/경로 설정 (디바이스명, 서비스/캐릭터리스틱 UUID, 저장 경로)
  2) FileReceiver: 단일 센서(ESP32)로부터 파일(META/DATA/EOF) 재조립
     - handle_meta()  : 전송될 파일의 메타(META) 수신/초기화
     - handle_data()  : 데이터 청크(DATA) 수신/EOF 판정
     - save_and_reset(): 수신 버퍼를 파일로 저장하고 상태 초기화
  3) device_worker(): 특정 이름의 BLE 디바이스를 스캔/연결/Notify 구독/재접속
  4) run_ble_receiver(): 여러 디바이스 워커를 동시에 실행
  5) run_listener_for_all(): 기존 main 코드 호환을 위한 thin adapter
  6) start_ble_receiver(): 동기 진입점(직접 실행 시)

[입출력 구조 예시]
  • 입력(Input):
    - BLE(ESP32)에서 전송되는 2개의 Notify 캐릭터리스틱
      (1) META: JSON 바이트 (예: {"id": 17, "size": 64000})
      (2) DATA: [seq_lo(1B), seq_hi(1B), flags(1B), payload(NB)]
          * seq   : 0부터 시작하는 16-bit 증가 시퀀스 번호
          * flags : bit0 == 1 이면 EOF(파일 전송 완료)
          * payload: WAV 파일 바이트의 조각
  • 처리(Process):
    - META 수신 시: 버퍼 초기화, 기대 파일 크기/ID 설정
    - DATA 수신 시: 순서 검사(seq), payload를 버퍼에 순차적으로 누적
    - EOF flag 감지 시: 누적 버퍼를 .wav 파일로 저장, 상태 리셋
  • 출력(Output):
    - 저장 경로: ./Input_data/real_input/<SensorName>_YYYYMMDD-HHMMSS.wav
    - 표준출력 로그: 연결/구독/수신 진행 상황과 경고 메시지

[운영 팁]
  - 디바이스명이 정확히 일치해야 발견/연결됩니다 (DEFAULT_DEVICE_NAMES 참조).
  - 연결이 끊기면 3초 간격으로 재스캔/재연결을 영속적으로 시도합니다.
  - 이 PC 코드는 *파일 저장만* 담당합니다. 임계치 감지/녹음/패키징은 ESP32가 수행합니다.
  - 경로는 상대 경로(이 파일이 위치한 폴더 기준)로 저장되며, 폴더가 없으면 생성합니다.

===============================================================================
"""

"""
# -----------------------------------------------------------------------------
# BLE-only multi-device WAV receiver
# - ESP32: (임계치 감지 → 2초 녹음 → WAV 패키징 → BLE 전송)
# - PC   : (BLE 연결 유지 + 파일 수신/저장)
#
# 기능 요약
#   1) 여러 ESP32(예: "Sensor-01", "Sensor-02", "Sensor-03")에 동시 접속 및 유지
#   2) 각 기기에서 META(JSON) → DATA(바이너리 청크) → EOF 순으로 수신하여 WAV 저장
#   3) 저장 경로: ./Input_data/real_input  (이 파일이 위치한 폴더 기준 상대 경로)
#   4) 끊김/에러 발생 시 자동 재검색/재연결 (무한 루프)
#
# 패킷 형식(ESP32 → PC):
#   - META 캐릭터리스틱(UUID: ...ef2): UTF-8 JSON e.g., {"id": 7, "size": 123456}
#   - DATA 캐릭터리스틱(UUID: ...ef3): [seq_lo(1B), seq_hi(1B), flags(1B), payload]
#       * seq   : 0부터 1씩 증가(16-bit)
#       * flags : bit0==1 → EOF(전송 종료), 기타 비트는 예약
#       * payload: 파일의 연속 바이트 조각
# -----------------------------------------------------------------------------
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import os
import argparse

from bleak import BleakScanner, BleakClient, BLEDevice

"""
This module can be imported and started from main.py via start_ble_receiver().
"""

# NOTE: 아래 이름과 UUID는 ESP32 펌웨어와 합의된 값이어야 합니다.
#       * 이름 일치 실패 시 스캔되더라도 연결을 시도하지 않습니다.
#       * UUID가 다르면 Notify 구독에 실패합니다.
DEFAULT_DEVICE_NAMES: List[str] = ["Sensor-01", "Sensor-02", "Sensor-03"]
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_FILE_META_UUID = "12345678-1234-5678-1234-56789abcdef2"
CHAR_FILE_DATA_UUID = "12345678-1234-5678-1234-56789abcdef3"

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_DIR = ROOT_DIR / "Input_data" / "real_input"
DEFAULT_TARGET_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Configuration (tunable)
# -------------------------
# 시간 파라미터(초) — 하드코딩된 상수 대신 사용
SCAN_TIMEOUT_S: float = float(os.getenv("BLE_SCAN_TIMEOUT_S", 5.0))
RESCAN_DELAY_S: float = float(os.getenv("BLE_RESCAN_DELAY_S", 3.0))
RECONNECT_DELAY_S: float = float(os.getenv("BLE_RECONNECT_DELAY_S", 3.0))

def parse_device_names(default: List[str]) -> List[str]:
    """환경변수 BLE_DEVICE_NAMES(쉼표 구분) 또는 기본값을 사용해 디바이스 명단 생성."""
    env = os.getenv("BLE_DEVICE_NAMES", "").strip()
    if not env:
        return default
    # 공백 제거 + 빈 항목 제거
    names = [x.strip() for x in env.split(",") if x.strip()]
    return names if names else default

# -------------------------
# 파일 수신 상태기
# -------------------------
class FileReceiver:
    """수신된 WAV 파일을 재조립/검증/저장하는 상태 머신.

    Responsibilities
    ----------------
    - META(JSON) 수신 시: 기대 파일 크기/ID 설정, 내부 버퍼 초기화
    - DATA 청크 수신 시: 시퀀스 검증 후 payload를 누적, EOF 감지
    - EOF 도달 시: 누적 버퍼를 .wav로 저장하고 상태 초기화

    Parameters
    ----------
    sensor_name : str
        파일명에 찍힐 센서(디바이스) 이름.
    save_dir : pathlib.Path
        저장 디렉터리 경로. 없으면 상위에서 생성.

    Notes
    -----
    - 시퀀스 번호가 건너뛰면 경고를 출력하지만, 가능한 한 수신을 계속합니다.
    - META가 동일 id로 중복 수신되면 무시하여 중복 저장을 방지합니다.
    - expected_size가 제공되면 저장 후 바이트 수를 검사합니다(로그 경고용).
    - 저장은 임시 확장자(.part)로 작성 후 원자적(rename)으로 교체하여 부분 파일 노출을 방지합니다.
    """
    def __init__(self, sensor_name: str, save_dir: Path):
        self.sensor_name = sensor_name
        self.save_dir = save_dir
        self.expected_size: Optional[int] = None
        self.current_id: Optional[int] = None
        self.buffer = bytearray()
        self.last_seq = -1
        self.ready = False

    def handle_meta(self, data: bytes):
        """META(JSON) 처리: 새로운 파일 전송을 준비한다."""
        # 1) UTF-8 JSON 파싱 (예: {"id": 17, "size": 64000})
        #    - id   : 전송 세션 식별자(동일 id 재전송 시 중복 저장 방지)
        #    - size : 전체 파일 바이트 수(선택적, 검증용)
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
        """DATA 청크 처리: 시퀀스 검증, payload 누적, EOF 감지."""
        # 데이터 구조: [seq_lo, seq_hi, flags, payload...]
        # - seq   : little-endian 16-bit
        # - flags : bit0==1 → EOF (이 청크로 전송 종료)
        # - payload: 실제 파일 바이트 조각

        # (안전장치) META 수신 전이라면 데이터는 폐기
        if not self.ready:
            print(f"[{self.sensor_name}] WARN: data before META, ignoring chunk")
            return False

        # (길이 검사) 최소 3바이트(seq_lo, seq_hi, flags) 필요
        if len(data) < 3:
            return False

        # 시퀀스 계산 및 순서 점검 (누락/재전송 감지용)
        seq = data[0] | (data[1] << 8)
        flags = data[2]
        payload = data[3:]

        if seq != self.last_seq + 1:
            if self.last_seq == -1:
                print(f"[{self.sensor_name}] INFO: first chunk seq={seq} (expected 0)")
            else:
                print(f"[{self.sensor_name}] WARN: unexpected seq {seq} (last {self.last_seq})")

        self.last_seq = seq

        # payload를 내부 버퍼에 누적
        self.buffer.extend(payload)

        # EOF 비트가 켜져 있으면 True 반환하여 상위에서 저장 트리거
        if flags & 0x01:  # EOF
            print(f"[{self.sensor_name}] EOF: total bytes={len(self.buffer)}")
            return True
        return False

    def save_and_reset(self) -> Path:
        """누적 버퍼를 .wav 파일로 저장하고 내부 상태를 초기화한다."""
        # 파일명 규칙: <센서명>_YYYYMMDD-HHMMSS.wav
        #   - 센서별/시간별 파일을 쉽게 구분 가능
        now = datetime.now()
        ts = now.strftime("%Y%m%d-%H%M%S") + f"-{now.microsecond:06d}"
        id_suffix = f"_{self.current_id}" if self.current_id is not None else ""
        filename = f"{self.sensor_name}{id_suffix}_{ts}.wav"
        path = self.save_dir / filename
        tmp_path = path.with_suffix(path.suffix + ".part")
        # 임시 파일로 먼저 기록 후 원자적으로 교체
        with open(tmp_path, "wb") as f:
            f.write(self.buffer)
        try:
            tmp_path.replace(path)
        except Exception:
            # 일부 파일시스템에서는 replace가 원자적이지 않을 수 있어 fallback
            os.replace(tmp_path, path)

        # 크기 검증: META.size가 제공되면 경고 수준으로만 비교
        if self.expected_size and len(self.buffer) != self.expected_size:
            print(f"[{self.sensor_name}] WARN: size mismatch (expected {self.expected_size}, got {len(self.buffer)})")

        # 상태 리셋: 다음 전송을 위해 모든 내부 상태 초기화
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
async def device_worker(device_name: str, save_dir: Path):
    """주어진 디바이스명을 목표로 스캔→연결→Notify 구독→수신을 반복.

    동작 순서
    --------
    1) 5초간 스캔하여 이름이 일치하는 디바이스 검색
    2) 발견 시 연결 후 META/DATA 캐릭터리스틱에 대해 start_notify()
    3) 콜백에서 FileReceiver로 전달하여 재조립
    4) 연결이 끊기거나 예외 발생 시 3초 후 재시도(무한 루프)

    Notes
    -----
    - 이 루틴은 영속적으로 동작하며 KeyboardInterrupt/취소 시 종료됩니다.
    - notify 콜백은 Bleak 스레드에서 호출되므로, 최소한의 작업만 수행합니다.
    """
    receiver = FileReceiver(sensor_name=device_name, save_dir=save_dir)

    while True:
        try:
            print(f"[{device_name}] Scanning for device...")
            dev: Optional[BLEDevice] = None
            devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT_S)
            for d in devices:
                if d.name == device_name:
                    dev = d
                    break

            if not dev:
                print(f"[{device_name}] Not found. Retry in 3s...")
                await asyncio.sleep(RESCAN_DELAY_S)
                continue

            print(f"[{device_name}] Found {dev.address}. Connecting...")
            async with BleakClient(dev) as client:
                print(f"[{device_name}] Connected.")

                def meta_cb(_, data: bytearray):
                    # Bleak가 bytearray를 전달하므로 불변 bytes로 변환해 파싱
                    # (파싱 중 참조가 유지되어도 안전)
                    receiver.handle_meta(bytes(data))

                def data_cb(_, data: bytearray):
                    # Bleak가 bytearray를 전달하므로 불변 bytes로 변환해 파싱
                    # (파싱 중 참조가 유지되어도 안전)
                    eof = receiver.handle_data(bytes(data))
                    if eof:
                        # EOF면 파일 저장을 트리거하고 상태 초기화
                        receiver.save_and_reset()

                # 두 캐릭터리스틱에 Notify 구독을 걸어 데이터 수신을 시작합니다.
                # 이후에는 콜백(meta_cb/data_cb)이 호출되며, 아래의 무한 sleep 루프는
                # 단순히 연결을 유지(작업 스레드 생존)하는 역할만 합니다.
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
            await asyncio.sleep(RECONNECT_DELAY_S)
            continue

# -------------------------
# BLE 수신기 실행 함수
# -------------------------
async def run_ble_receiver(device_names: Optional[List[str]] = None, save_dir: Optional[Path] = None):
    """여러 디바이스 워커를 동시 실행하는 메인 async 함수."""
    # 이름 리스트에 대해 worker 태스크를 생성하고 gather로 영구 대기
    names = device_names or parse_device_names(DEFAULT_DEVICE_NAMES)
    out_dir = save_dir or DEFAULT_TARGET_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [asyncio.create_task(device_worker(name, out_dir)) for name in names]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\n[PC] Stopping...")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# -----------------------------------------------------------------------------
# Compatibility wrapper for existing main code
# main_no_socket.py expects run_listener_for_all(real_time_folder=...)
# This thin adapter forwards to start_ble_receiver(save_dir=...)
# -----------------------------------------------------------------------------
def run_listener_for_all(real_time_folder: str = None):
    """기존 main 코드의 run_listener_for_all(real_time_folder=...)과 호환되는 래퍼."""
    # Windows에서 Bleak 이벤트루프 정책이 필요할 수 있어 선설정
    # run_ble_receiver를 동기적으로 실행
    save_dir = Path(real_time_folder) if real_time_folder else DEFAULT_TARGET_DIR
    device_names = parse_device_names(DEFAULT_DEVICE_NAMES)
    print("[PC] Device list:", ", ".join(device_names))
    # Windows 환경에서 asyncio 루프 정책 설정(필요 시)
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    # 동기 진입
    asyncio.run(run_ble_receiver(device_names=device_names, save_dir=save_dir))

def start_ble_receiver(device_names: Optional[List[str]] = None, save_dir: Optional[Path] = None):
    """동기 컨텍스트에서 BLE 수신기를 즉시 실행하는 편의 함수."""
    asyncio.run(run_ble_receiver(device_names, save_dir))

if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="BLE multi-device WAV receiver")
    parser.add_argument("--devices", type=str, default=os.getenv("BLE_DEVICE_NAMES", ""),
                        help="Comma-separated device names (overrides env BLE_DEVICE_NAMES).")
    parser.add_argument("--out", type=str, default=str(DEFAULT_TARGET_DIR),
                        help="Output directory for received WAV files.")
    args = parser.parse_args()

    # 디바이스 해석 우선순위: --devices CLI > BLE_DEVICE_NAMES env > DEFAULT_DEVICE_NAMES
    if args.devices.strip():
        names = [x.strip() for x in args.devices.split(",") if x.strip()]
    else:
        names = parse_device_names(DEFAULT_DEVICE_NAMES)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[PC] Target save dir:", out_dir)
    print("[PC] Device list:", ", ".join(names))
    start_ble_receiver(device_names=names, save_dir=out_dir)
