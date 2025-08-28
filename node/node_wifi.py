"""
===============================================================================
Wi‑Fi WAV Receiver (PC-side) — HTTP Server
===============================================================================
이 모듈은 ESP32가 Wi‑Fi(HTTP POST)로 업로드한 WAV 데이터를 받아
디스크에 저장하는 **PC 수신기**입니다. 기존 BLE 수신기(node_BLE.py)를
HTTP 방식으로 대체한 버전이며, 파일 저장 경로/호출 인터페이스는 가능한
범위에서 기존과 호환되도록 유지했습니다.

[사전 준비]
  - Python 3.9+ 권장
  - 패키지 설치:  pip install flask
  - 방화벽 예외:  5050/tcp (기본값) 포트를 수신 가능하게 설정

[실행 방법]
  1) 개발 서버(Flask)로 실행:
     python node_wifi.py --out ./Input_data/real_input --host 0.0.0.0 --port 5050
     - 실서비스에서는 Flask 개발 서버 대신 gunicorn/uwsgi 사용 권장
       예) gunicorn -w 2 -b 0.0.0.0:5050 node_wifi:create_app
           (환경에 따라 PYTHONPATH 조정 필요)

[ESP32 → 서버 업로드 프로토콜]
  - 요청(HTTP POST /upload)
    Headers:
      X-Room-ID    : 방/구역 이름(예: Room102)           [선택, 기본 "unknown"]
      X-Mic-ID     : 센서(마이크) 이름(예: Sensor-01)     [선택, 기본 "unknown"]
      X-Timestamp  : epoch seconds(정수/실수)             [선택]
                      미제공 또는 파싱 실패 시 서버 수신 시각 사용
    Body:
      WAV 바이트(헤더 포함). Content-Type은 application/octet-stream 권장.

  - 응답(JSON):
      { "ok": true, "saved": "<절대경로>", "size": <바이트수> }
      실패 시: HTTP 4xx/5xx + { "ok": false, "error": "<원인>" }

[저장 규칙]
  - 상대경로 기준(이 파일이 위치한 폴더): ./Input_data/real_input
  - 파일명: <MicID>_<RoomID>_<YYYYMMDD-HHMMSS-uuuuuu>.wav
    * MicID/RoomID는 안전문자만 허용(영숫자, '-', '_') → 경로침투 방지
    * 저장 시 먼저 .part 확장자로 임시 파일을 만든 뒤, 원자적으로 교체

[운영 팁]
  - 다중 ESP32의 동시 업로드 대응: 요청마다 독립 처리되므로 자연스럽게 병렬 처리됨
  - 헬스체크: GET /health → {"ok": true}
  - 디스크 용량/권한 확인: 저장 실패가 잦다면 폴더 권한 및 여유 용량 점검
  - 보안(외부망 노출 시):
      • HTTPS(리버스 프록시) + API 토큰/서명 헤더 검증
      • 파일 크기 제한, IP 화이트리스트, 인증 미들웨어 등 추가 고려

===============================================================================
"""

from __future__ import annotations

# 표준 라이브러리
import os
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# 외부 라이브러리
# - Flask: 간단한 HTTP 서버/라우팅 프레임워크
from flask import Flask, request, jsonify

# -----------------------------------------------------------------------------
# 저장 경로 기본값 (기존 상대 경로 유지)
#   - ROOT_DIR : 이 파일(node_wifi.py)이 위치한 디렉토리
#   - DEFAULT_TARGET_DIR : 기본 저장 디렉토리 (없으면 자동 생성)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TARGET_DIR = ROOT_DIR / "Input_data" / "real_input"
DEFAULT_TARGET_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Flask 애플리케이션 팩토리
#   - save_dir: 업로드된 WAV 파일을 저장할 대상 디렉터리
#   - 반환값  : 구성된 Flask 앱 객체
# -----------------------------------------------------------------------------
def create_app(save_dir: Path) -> Flask:
    app = Flask(__name__)
    # 요청 시점에 디렉터리가 없을 수도 있으므로 재확인하여 생성
    save_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/health")
    def health():
      """
      헬스체크 엔드포인트.
      - 로드밸런서/L7 헬스체크, 수동 점검 등에 사용
      - 간단히 200 OK + {"ok": true} 반환
      """
      return jsonify(ok=True)

    @app.post("/upload")
    def upload():
        """
        ESP32가 업로드하는 WAV 수신 엔드포인트.

        요청 규약(권장):
          Content-Type: application/octet-stream
          Headers:
            X-Room-ID    : 방/구역 식별자(문자/숫자/대시/언더스코어만 사용 권장)
            X-Mic-ID     : 디바이스/마이크 식별자(동일)
            X-Timestamp  : epoch seconds(정수/실수). 없으면 서버 수신 시각 사용.

        동작:
          1) 헤더 파싱 및 값 정규화(파일명 안전 문자만 유지)
          2) 본문(raw body)을 읽어 .part 임시파일에 기록
          3) 기록 완료 후 원자적 교체(rename/replace)로 .wav 확정
        실패:
          - 본문이 비어있을 경우 400
          - 저장 중 예외 발생 시 500
        """
        # --- 1) 메타 정보 파싱 (헤더는 필수가 아님) -------------------------
        room = request.headers.get("X-Room-ID", "unknown")
        mic  = request.headers.get("X-Mic-ID", "unknown")
        ts_h = request.headers.get("X-Timestamp", "")

        # --- 2) 타임스탬프 처리 --------------------------------------------
        #  - epoch seconds로 전달되면 우선 사용
        #  - 문자열 파싱 실패 시 서버 현재 시각으로 대체
        ts_dt: datetime
        if ts_h:
            try:
                ts_dt = datetime.fromtimestamp(float(ts_h), tz=timezone.utc).astimezone()
            except Exception:
                ts_dt = datetime.now()
        else:
            ts_dt = datetime.now()

        # --- 3) 안전한 파일명 생성 -----------------------------------------
        #  - MicID/RoomID 는 영숫자/하이픈/언더스코어만 허용하여 경로침투 방지
        #  - 마이크로초 단위까지 포함해 충돌 확률 최소화
        ts_str = ts_dt.strftime("%Y%m%d-%H%M%S") + f"-{ts_dt.microsecond:06d}"
        safe_room = "".join(c for c in room if c.isalnum() or c in ("-", "_"))
        safe_mic  = "".join(c for c in mic  if c.isalnum() or c in ("-", "_"))
        fname = f"{safe_mic}_{safe_room}_{ts_str}.wav"
        out_path = save_dir / fname
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")

        # --- 4) 본문(raw body) 추출 -----------------------------------------
        #  - Flask는 request.get_data()로 전체 바디 바이트를 제공
        #  - 매우 큰 파일에 대해서는 스트리밍 처리/사이즈 제한을 고려할 것
        data = request.get_data(cache=False, as_text=False)
        if not data:
            return jsonify(ok=False, error="empty body"), 400

        # --- 5) 안전 저장 (.part → 원자적 교체) ------------------------------
        #  - .part 임시 파일에 먼저 쓰고, replace/rename 으로 완성본 교체
        #  - 전원 장애/예외 발생 시 부분 파일 노출 방지
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
            # 파일 교체(플랫폼별 원자성 보장 수준이 다를 수 있어 두 방식 모두 시도) 
            try:
                tmp_path.replace(out_path)  # Python 3.8+: Path.replace
            except Exception:
                os.replace(tmp_path, out_path)
        except Exception as e:
            # 실패 시 .part 파일 정리 시도(있다면 삭제)
            try:
                if tmp_path.exists():
                    # Python 3.8+: unlink(missing_ok=True) 지원.
                    # 하위 버전 호환을 위해 예외 무시.
                    tmp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            return jsonify(ok=False, error=str(e)), 500

        # --- 6) 완료 로그 및 응답 -------------------------------------------
        print(f"[HTTP] Saved: {out_path}")
        return jsonify(ok=True, saved=str(out_path), size=len(data))

    return app

# -----------------------------------------------------------------------------
# 기존 main 코드 호환 래퍼
#   - 이전 코드에서 run_listener_for_all(real_time_folder=...)를 호출하던
#     부분을 그대로 사용할 수 있도록 thin wrapper를 유지
#   - 개발 편의를 위해 Flask 개발 서버를 직접 띄우지만,
#     프로덕션에서는 gunicorn/uwsgi 사용 권장
# -----------------------------------------------------------------------------
def run_listener_for_all(real_time_folder: str = None,
                         host: str = "0.0.0.0",
                         port: int = 5050):
    """
    기존 main 코드의 run_listener_for_all(real_time_folder=...)과 호환되는 래퍼.
    BLE 수신기 대신 HTTP 서버를 실행합니다.

    Parameters
    ----------
    real_time_folder : str | None
        업로드 파일 저장 폴더. None이면 DEFAULT_TARGET_DIR 사용.
    host : str
        바인딩할 인터페이스(IP). 기본 0.0.0.0(모든 NIC).
    port : int
        수신 포트. 기본 5050.
    """
    save_dir = Path(real_time_folder) if real_time_folder else DEFAULT_TARGET_DIR
    app = create_app(save_dir)
    # Flask 개발 서버 (단일 프로세스/스레드). 운영환경은 WSGI 서버 권장.
    app.run(host=host, port=port, debug=True)

# -----------------------------------------------------------------------------
# 동기 진입점 (라이브러리처럼 import하지 않고 직접 실행할 때)
# -----------------------------------------------------------------------------
def start_wifi_receiver(save_dir: Optional[Path] = None,
                        host: str = "0.0.0.0",
                        port: int = 5050):
    """
    간단 실행용 진입점 헬퍼.
    """
    save_dir = save_dir or DEFAULT_TARGET_DIR
    run_listener_for_all(real_time_folder=str(save_dir), host=host, port=port)

# -----------------------------------------------------------------------------
# CLI 엔트리포인트
#   예)
#     python node_wifi.py --out ./Input_data/real_input --host 0.0.0.0 --port 5050
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wi‑Fi HTTP WAV receiver")
    parser.add_argument("--out", type=str, default=str(DEFAULT_TARGET_DIR),
                        help="수신 WAV 저장 디렉터리")
    parser.add_argument("--host", type=str, default=os.getenv("HTTP_HOST", "0.0.0.0"),
                        help="바인딩 호스트 IP (기본: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("HTTP_PORT", "5050")),
                        help="수신 포트 (기본: 5050)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[HTTP] Target save dir:", out_dir)

    # Flask 개발 서버 실행 (운영환경에서는 WSGI 서버 사용 권장)
    run_listener_for_all(real_time_folder=str(out_dir), host=args.host, port=args.port)
