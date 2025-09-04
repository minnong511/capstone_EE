#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6. 메인 루프 (main())
	1.	폴더 스캔: 새 wav 파일 확인
	2.	이벤트 그룹화: 시간 차이가 1초 이내인 파일 묶음
	3.	선택: 이벤트 중 peak dBFS가 가장 큰 것 하나 선택
	4.	전처리 및 TDOA: 신호 로드 → 필터링 → 시간차 계산
	5.	좌표 추정: 격자 탐색으로 (x, y) 계산
	6.	출력/저장: 콘솔 로그 + JSON 파일 저장
"""

"""
[슬라이딩 윈도우 TDOA 러너 - 샘플 구조]

INPUT
- 디렉토리: ./Input_data/real_input
- 매 2초마다 새로 들어온 wav 파일 스캔
- 1초 이내 생성된 파일들을 '이벤트'로 묶음
- 이벤트들 중 RMS dBFS(최대값)가 가장 큰 '하나'만 선택

처리조건
- 선택된 이벤트에 포함된 다중 마이크 wav를 사용해 GCC-PHAT TDOA
- 기준 마이크 ref_id는 가장 많은/혹은 지정 ID 사용
- 밴드패스/zero-mean 등 최소 전처리

OUTPUT
- 콘솔: 추정 좌표 (x_hat, y_hat)
- 파일(옵션): results/tdoa_YYYYmmdd_HHMMSS.json 저장

※ 이 코드는 '구조 샘플'입니다. 프로젝트의 tdoa.py 모듈이 있다면 해당 함수들을 import하여 교체하세요.
"""

import time
import json
import shutil
import math
import fnmatch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import soundfile as sf
    HAVE_SF = True
except Exception:
    import wave
    HAVE_SF = False

# --------------------------
# 설정 파라미터
# --------------------------
ROOT_DIR = Path(__file__).resolve().parent  # 이 파일 위치 기준
IN_DIR = ROOT_DIR / "Input_data" / "real_input_tdoa"
# 가장 큰 소리를 복사할 목적지 디렉토리 (슬라이딩 윈도우 이후)
DEST_DIR = ROOT_DIR / "Input_data" / "real_input"
DEST_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR = ROOT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SCAN_INTERVAL_SEC = 2.0           # 슬라이딩 윈도우 주기
EVENT_GROUP_WINDOW_SEC = 1.0      # 이벤트 묶음 허용 시간 간격
MIN_MICS_REQUIRED = 3             # TDOA 최소 마이크 수(2D 권장 3개 이상)
FS_TARGET = 16000                 # 리샘플 목표 fs (필요 시)
BANDPASS = (300.0, 5000.0)        # Hz (옵션: 원치 않으면 None)
SPEED_OF_SOUND = 343.0            # m/s

# 마이크 ID → 좌표 (m)
MIC_XY = {
    "Sensor-01": (0.0, 0.0),
    "Sensor-02": (2.0, 0.0),
    "Sensor-03": (0.0, 2.0),
}
REF_ID = "Sensor-01"  # 기준 마이크 (없으면 자동 선택)

# 그리드 탐색 영역 (m)
GRID = (-1.0, 3.0, -1.0, 3.0, 0.05)  # xmin, xmax, ymin, ymax, step

# --------------------------
# 유틸
# --------------------------

def list_wavs_since(dirpath: Path, since_ts: float) -> List[Path]:
    """since_ts(UNIX epoch) 이후 생성/수정된 wav 목록."""
    files = []
    for p in dirpath.glob("*.wav"):
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime > since_ts:
            files.append(p)
    return sorted(files, key=lambda x: x.stat().st_mtime)

def parse_mic_id_from_name(path: Path) -> Optional[str]:
    """파일명에서 센서ID 추출 (패턴이 명확하지 않아도 'Sensor-xx' 서브스트링으로 탐색)."""
    name = path.name
    for mic_id in MIC_XY.keys():
        if mic_id in name:
            return mic_id
    # fallback: Sensor-숫자 패턴
    tokens = ["Sensor-01","Sensor-02","Sensor-03","Sensor-04"]
    for t in tokens:
        if t in name:
            return t
    return None

def load_wav_mono(path: Path, fs_target: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """wav 로드 → mono float32, 필요 시 간단 리샘플."""
    if HAVE_SF:
        data, fs = sf.read(str(path), always_2d=True)
        x = data.mean(axis=1).astype(np.float32)
    else:
        with wave.open(str(path), "rb") as w:
            fs = w.getframerate()
            n = w.getnframes()
            raw = w.readframes(n)
            ch = w.getnchannels()
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)

    if fs_target and fs_target != fs:
        # 간단한 리샘플(고급 필요 시 scipy.signal.resample_poly 사용)
        rat = fs_target / fs
        tgt_len = int(round(len(x) * rat))
        x = np.interp(np.linspace(0, len(x), tgt_len, endpoint=False), np.arange(len(x)), x).astype(np.float32)
        fs = fs_target
    return x, fs

def zero_mean(x: np.ndarray) -> np.ndarray:
    m = float(np.mean(x)) if x.size else 0.0
    return (x - m).astype(np.float32)

def butter_bandpass(band: Tuple[float,float], fs: int, order=4):
    from math import tan, pi
    from scipy.signal import butter
    low, high = band
    return butter(order, [low/(fs*0.5), high/(fs*0.5)], btype="band")

def apply_bandpass(x: np.ndarray, fs: int, band: Tuple[float,float]) -> np.ndarray:
    from scipy.signal import lfilter
    b,a = butter_bandpass(band, fs)
    return lfilter(b, a, x).astype(np.float32)

def rms_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)

# --------------------------
# GCC-PHAT TDOA (간단 버전)
# --------------------------

def gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: int, interp: int = 16, max_tau: Optional[float]=None) -> float:
    """sig vs ref 시간차(초)."""
    n = 1
    L = len(sig) + len(ref)
    while n < L:
        n <<= 1
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)
    R = SIG * np.conj(REF)
    denom = np.abs(R) + 1e-12
    G = R / denom
    cc = np.fft.irfft(G, n=interp*n)
    cc = np.concatenate((cc[-(interp*len(sig))//2:], cc[:(interp*len(sig))//2]))
    shift = np.argmax(np.abs(cc)) - len(cc)//2

    if max_tau is not None:
        max_shift = int(max_tau * fs * interp)
        shift = int(np.clip(shift, -max_shift, max_shift))
    tau = shift / (fs * interp)
    return float(tau)

def tdoa_estimate(signals: Dict[str, np.ndarray], fs: int, ref_id: str, mic_xy: Dict[str, Tuple[float,float]]) -> Dict[str, float]:
    """각 마이크의 ref 대비 지연(초)."""
    if ref_id not in signals:
        # ref 없으면 임의 선택
        ref_id = sorted(signals.keys())[0]
    ref = signals[ref_id]
    taus = {ref_id: 0.0}
    # 최대 지연(물리 한계): 마이크 간 최대 거리
    coords = np.array(list(mic_xy.values()), dtype=np.float32)
    dmax = 0.0
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dmax = max(dmax, float(np.linalg.norm(coords[i]-coords[j])))
    max_tau = dmax / SPEED_OF_SOUND if dmax > 0 else None

    for mid, x in signals.items():
        if mid == ref_id:
            continue
        taus[mid] = gcc_phat(x, ref, fs, interp=16, max_tau=max_tau)
    return taus

def grid_localize(taus_obs: Dict[str,float], ref_id: str, mic_xy: Dict[str,Tuple[float,float]], grid: Tuple[float,float,float,float,float]) -> Tuple[float,float,float]:
    """L1 비용으로 2D 위치 추정. return: (x_hat, y_hat, min_cost)."""
    xmin,xmax,ymin,ymax,step = grid
    xs = np.arange(xmin, xmax+1e-9, step, dtype=np.float32)
    ys = np.arange(ymin, ymax+1e-9, step, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    cost = np.zeros_like(X, dtype=np.float32)

    xr, yr = mic_xy[ref_id]
    for mid, tau_obs in taus_obs.items():
        if mid == ref_id:
            continue
        xi, yi = mic_xy[mid]
        di = np.sqrt((X - xi)**2 + (Y - yi)**2)
        dr = np.sqrt((X - xr)**2 + (Y - yr)**2)
        tau_pred = (di - dr) / SPEED_OF_SOUND
        cost += np.abs(tau_pred - tau_obs).astype(np.float32)

    idx = np.argmin(cost)
    y_idx, x_idx = np.unravel_index(idx, cost.shape)
    return float(xs[x_idx]), float(ys[y_idx]), float(cost[y_idx, x_idx])

# --------------------------
# 이벤트 구성 & 선택
# --------------------------

@dataclass
class Event:
    key_ts: float                      # 이벤트 대표 시각(초)
    files: Dict[str, Path]             # mic_id -> 파일 경로
    peak_db: float                     # 이벤트 내 파일들 중 최대 dBFS
    peak_file: Path | None = None

def group_events(files: List[Path], window_sec: float) -> List[Event]:
    """파일들을 시간 기준으로 묶어 이벤트 리스트를 만든다."""
    events: List[Event] = []
    if not files:
        return events

    # 시간 순으로 정렬
    files = sorted(files, key=lambda p: p.stat().st_mtime)
    current_group: Dict[str, Path] = {}
    group_times: List[float] = []

    def finalize_group():
        if not current_group:
            return
        # 그룹 peak dB 계산 (파일 하나씩 로드 → RMS dBFS 최대)
        peak = -120.0
        peak_path: Path | None = None
        for p in current_group.values():
            x, fs = load_wav_mono(p, FS_TARGET)
            x = zero_mean(x)
            if BANDPASS:
                try:
                    x = apply_bandpass(x, fs, BANDPASS)
                except Exception:
                    pass
            val = rms_dbfs(x)
            if val > peak:
                peak = val
                peak_path = p
        ev = Event(key_ts=float(np.mean(group_times)), files=dict(current_group), peak_db=peak, peak_file=peak_path)
        events.append(ev)

    base_time = files[0].stat().st_mtime
    for p in files:
        mt = p.stat().st_mtime
        if (mt - base_time) <= window_sec:
            mic_id = parse_mic_id_from_name(p) or p.stem.split("_")[0]
            current_group[mic_id] = p
            group_times.append(mt)
        else:
            finalize_group()
            # 새 그룹 시작
            current_group = {}
            group_times = []
            base_time = mt
            mic_id = parse_mic_id_from_name(p) or p.stem.split("_")[0]
            current_group[mic_id] = p
            group_times.append(mt)

    finalize_group()
    return events

# --------------------------
# 공개 API: 1회 실행/처리 유닛
# --------------------------

def _prepare_signals(available: Dict[str, Path]) -> Tuple[Dict[str, np.ndarray], int]:
    """마이크 wav 경로 dict -> (전처리된 신호 dict, fs) 반환"""
    signals: Dict[str, np.ndarray] = {}
    fs_used: Optional[int] = None
    for mid, p in available.items():
        x, fs = load_wav_mono(p, FS_TARGET)
        x = zero_mean(x)
        if BANDPASS:
            try:
                x = apply_bandpass(x, fs, BANDPASS)
            except Exception:
                pass
        signals[mid] = x
        fs_used = fs if fs_used is None else fs_used
    if fs_used is None:
        fs_used = FS_TARGET
    return signals, fs_used


def _process_event(ev: 'Event', dest_dir: Path = DEST_DIR, result_dir: Path = RESULT_DIR) -> Optional[dict]:
    """단일 이벤트 처리: TDOA -> 좌표 추정 -> 결과 저장/복사. 결과 dict 반환."""
    # 등록된 MIC_XY에 있는 마이크만 사용
    available = {mid: path for mid, path in ev.files.items() if mid in MIC_XY}
    if len(available) < MIN_MICS_REQUIRED:
        print(f"[Runner] skip (mics<{MIN_MICS_REQUIRED}). have: {list(available.keys())}")
        return None

    # 신호 준비
    signals, fs_used = _prepare_signals(available)

    # 기준 마이크 결정
    ref = REF_ID if REF_ID in signals else list(signals.keys())[0]

    # TDOA -> 좌표
    taus = tdoa_estimate(signals, fs_used, ref, MIC_XY)
    x_hat, y_hat, cmin = grid_localize(taus, ref, MIC_XY, GRID)
    print(f"[TDOA] Estimated position: x={x_hat:.2f} m, y={y_hat:.2f} m (cost={cmin:.3g})")

    # 결과 구성 & 저장
    out = {
        "ts": ev.key_ts,
        "peak_db": ev.peak_db,
        "files": {k: str(v) for k, v in available.items()},
        "taus": taus,
        "x_hat": x_hat, "y_hat": y_hat, "cost": cmin,
        "grid": GRID, "bandpass": BANDPASS, "fs": fs_used,
    }
    out_path = result_dir / f"tdoa_{time.strftime('%Y%m%d_%H%M%S', time.localtime(ev.key_ts))}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[TDOA] result saved: {out_path}")

    # 가장 큰 소리 파일 복사
    try:
        if ev.peak_file and ev.peak_file.exists():
            ts_tag = time.strftime('%Y%m%d-%H%M%S', time.localtime(ev.key_ts))
            dst_name = f"{ts_tag}__{ev.peak_file.name}"
            dst_path = dest_dir / dst_name
            shutil.copy2(ev.peak_file, dst_path)
            print(f"[COPY] loudest file copied to: {dst_path}")
        else:
            print("[COPY] peak file not found; skip copy.")
    except Exception as ce:
        print(f"[COPY][ERROR] {ce}")

    return out


def run_tdoa_once(in_dir: Path = IN_DIR, dest_dir: Path = DEST_DIR, result_dir: Path = RESULT_DIR,
                   event_window: float = EVENT_GROUP_WINDOW_SEC) -> Optional[dict]:
    """한 번만 스캔하여 이벤트가 있으면 가장 큰 이벤트를 처리하고 결과 dict를 반환한다."""
    new_files = [p for p in list_wavs_since(in_dir, 0.0) if p.exists()]
    if not new_files:
        return None
    events = group_events(new_files, event_window)
    if not events:
        return None

    ev = max(events, key=lambda e: e.peak_db)
    print(f"[Runner] picked event @ {time.strftime('%H:%M:%S', time.localtime(ev.key_ts))} peak={ev.peak_db:.1f} dBFS, files={len(ev.files)}")
    result = _process_event(ev, dest_dir=dest_dir, result_dir=result_dir)
    return result

# --------------------------
# 메인 루프 (루프형 공개 API)
# --------------------------

def run_tdoa_loop(in_dir: Path = IN_DIR, dest_dir: Path = DEST_DIR, result_dir: Path = RESULT_DIR,
                  scan_interval: float = SCAN_INTERVAL_SEC, event_window: float = EVENT_GROUP_WINDOW_SEC) -> None:
    print(f"[Runner] watching: {in_dir}")
    last_scan_ts = 0.0
    seen_files: set[str] = set()

    while True:
        try:
            new_files = [p for p in list_wavs_since(in_dir, last_scan_ts) if p.exists()]
            if new_files:
                last_scan_ts = max(p.stat().st_mtime for p in new_files)
                new_files = [p for p in new_files if str(p) not in seen_files]
                for p in new_files:
                    seen_files.add(str(p))

                events = group_events(new_files, event_window)
                if events:
                    ev = max(events, key=lambda e: e.peak_db)
                    print(f"[Runner] picked event @ {time.strftime('%H:%M:%S', time.localtime(ev.key_ts))} peak={ev.peak_db:.1f} dBFS, files={len(ev.files)}")
                    _process_event(ev, dest_dir=dest_dir, result_dir=result_dir)

            time.sleep(scan_interval)

        except KeyboardInterrupt:
            print("\n[Runner] Stop.")
            break
        except Exception as e:
            print(f"[Runner][ERROR] {e}")
            time.sleep(scan_interval)


def main():
    IN_DIR.mkdir(parents=True, exist_ok=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_tdoa_loop()

if __name__ == "__main__":
    IN_DIR.mkdir(parents=True, exist_ok=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    main()