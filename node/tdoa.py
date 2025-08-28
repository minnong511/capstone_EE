#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TDOA(도달 시간 차) 기반 2D 위치 추정 유틸리티 — 파일 입력 버전
===============================================================================
[목표]
  여러 개의 마이크로 동시에 녹음된 WAV 파일들을 읽어들여, 마이크 쌍 간
  "도달 시간 차(Time Difference of Arrival; TDOA)"를 GCC‑PHAT 방법으로
  추정하고, 주어진 마이크 좌표(2D 평면) 위에서 격자(Grid) 탐색을 통해
  음원의 위치를 추정(heatmap + 최솟값 좌표)합니다.

[INPUT]
  1) WAV 파일들(.wav)
     - 각 마이크에 해당하는 파일 1개씩(또는 패턴으로 여러 후보 중 최신 1개 선택)
     - 모노 또는 스테레오(스테레오면 자동으로 모노 평균)
     - 샘플레이트는 모두 동일(다르면 --fs 로 강제 리샘플)
     - 파일명에 마이크 ID 문자열이 포함되어 있다고 가정(자동 매칭용)

  2) 마이크 좌표(--mic)
     - 형식: "<MicID>:<x>,<y>" (예: "Sensor-01:0,0")  단위: 미터(m)
     - 마이크 ID는 파일명에 포함된 문자열과 동일해야 매칭됩니다.

  3) 기준 마이크(--ref)
     - 모든 지연은 ref 대비로 계산됩니다(기준 지연 = 0).

  4) 격자 범위/해상도(--grid)
     - 형식: xmin xmax ymin ymax step (단위 m)
     - 예: --grid -1 3 -1 3 0.02  →  x∈[-1,3], y∈[-1,3], 격자 간격 0.02m

  5) (선택) 대역통과 필터(--band)
     - 형식: F_LO F_HI (Hz)
     - 예: --band 300 5000  → 음성/환경 소리 잡음을 줄일 때 유용

[처리 조건 / 가정]
  • 동시 녹음(시간 동기)이 되어 있어야 TDOA 신뢰도가 높습니다.
  • 신호 길이는 자동으로 최소 길이에 맞춰 잘립니다.
  • DC 성분 제거(zero‑mean) 기본 적용, 필요 시 band‑pass 추가 적용 가능.
  • TDOA 추정은 FFT 영역에서 PHAT 가중치를 적용한 GCC‑PHAT을 사용합니다.
  • 격자 탐색의 비용 함수는 Σ|τ_pred(x,y) − τ_obs| (각 마이크쌍에 대해)
    - τ_pred(x,y): 후보점(x,y) 기준의 이론적 지연((|P−Mi|−|P−Mr|)/c)
    - τ_obs: GCC‑PHAT으로 관측된 지연(초)
  • 음속 c 기본값 343 m/s (20°C 근처 공기). 필요 시 --c로 조정하세요.
  • --maxdist 를 주면 GCC‑PHAT에서 허용되는 최대 지연을 물리 거리로 제한
    (잡음/오검출 완화에 유리).

[OUTPUT]
  • 표준 출력
    - 추정 위치 (x_hat, y_hat) [m]
  • (옵션) 시각화(--show)
    - heatmap(낮을수록 좋음), 마이크 위치(▲), 추정점(x) 표시

[사용 예]
  python tdoa.py \
    --files ./Input_data/real_input/Sensor-01_*.wav \
            ./Input_data/real_input/Sensor-02_*.wav \
            ./Input_data/real_input/Sensor-03_*.wav \
    --mic "Sensor-01:0,0" "Sensor-02:2.0,0" "Sensor-03:0,2.0" \
    --ref Sensor-01 \
    --band 300 5000 \
    --grid -1 3 -1 3 0.02 \
    --show

[의존성]
  pip install numpy scipy matplotlib soundfile
  • soundfile이 없으면 표준 모듈 wave 로 폴백하여 로딩합니다(16bit PCM 가정).
===============================================================================
"""
from __future__ import annotations

# 표준 라이브러리
import argparse
import glob
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 서드파티
import numpy as np
import scipy.signal as sps

# --------- WAV 로더: soundfile 사용 우선, 실패 시 wave 폴백 ----------
try:
    import soundfile as sf  # type: ignore

    def load_wav_mono(path: str, target_fs: int | None = None) -> tuple[np.ndarray, int]:
        """WAV를 로드하여 mono float32로 반환. 필요 시 리샘플링.
        - path: 파일 경로
        - target_fs: 지정 시 해당 샘플레이트로 리샘플
        반환: (mono_signal[float32], fs[int])
        """
        data, fs = sf.read(path, always_2d=False)
        # 스테레오/멀티채널이면 평균으로 모노 변환
        if data.ndim == 2:
            data = data.mean(axis=1)
        # float32 캐스팅(정규화 가정)
        data = data.astype(np.float32, copy=False)
        # 필요 시 리샘플
        if target_fs and fs != target_fs:
            data = resample_kaiser(data, fs, target_fs)
            fs = target_fs
        return data, fs
except Exception:
    import wave

    def load_wav_mono(path: str, target_fs: int | None = None) -> tuple[np.ndarray, int]:
        """표준 wave 모듈 폴백 로더 (16bit PCM 가정).
        - 채널>1이면 평균 모노화
        - 필요 시 리샘플
        반환: (mono_signal[float32], fs[int])
        """
        with wave.open(path, "rb") as w:
            fs = w.getframerate()
            ch = w.getnchannels()
            n = w.getnframes()
            raw = w.readframes(n)
        # int16 → float32 정규화
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            x = x.reshape(-1, ch).mean(axis=1)
        if target_fs and fs != target_fs:
            x = resample_kaiser(x, fs, target_fs)
            fs = target_fs
        return x, fs

# -----------------------------------------------------------------------------
# 신호 처리 유틸리티
# -----------------------------------------------------------------------------

def resample_kaiser(x: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    """Kaiser window 기반 polyphase resampling (scipy.resample_poly).
    - 샘플레이트 정수비를 자동으로 약분(GCD)하여 up/down 비율 계산
    - 반환: float32
    """
    g = math.gcd(fs, target_fs)
    up, down = target_fs // g, fs // g
    return sps.resample_poly(x, up=up, down=down, window=("kaiser", 5.0)).astype(np.float32)


def bandpass_iir(x: np.ndarray, fs: int, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    """Butterworth 대역통과 필터.
    - f_lo~f_hi(Hz) 범위를 통과
    - 기본 차수 4 (필요 시 변경 가능)
    """
    ny = 0.5 * fs
    lo = max(1e-3, f_lo / ny)
    hi = min(0.999, f_hi / ny)
    b, a = sps.butter(order, [lo, hi], btype="band")
    return sps.lfilter(b, a, x).astype(np.float32)


def zero_mean(x: np.ndarray) -> np.ndarray:
    """DC 제거(평균 0으로 이동)."""
    x = x - np.mean(x)
    return x.astype(np.float32)


# -----------------------------------------------------------------------------
# GCC‑PHAT: 주파수 영역 상호상관 기반의 견고한 지연 추정
# -----------------------------------------------------------------------------

def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    fs: int,
    max_tau: float | None = None,
    interp: int = 16,
) -> Tuple[float, np.ndarray]:
    """두 신호(sig, ref) 사이의 시간 지연(tau)을 GCC‑PHAT으로 추정합니다.

    Parameters
    ----------
    sig, ref : np.ndarray
        비교할 두 신호(동일 샘플레이트)
    fs : int
        샘플레이트(Hz)
    max_tau : float | None
        허용 최대 지연(초). 지정 시 추정치를 클리핑하여 물리적으로 타당한 범위로 제한.
    interp : int
        상호상관 보간 배수(해상도 향상). 기본 16.

    Returns
    -------
    tau : float
        추정 지연(초). 양수면 sig가 ref보다 늦게 도달.
    cc : np.ndarray
        보간된 상호상관 파형(진단/시각화용).
    """
    # FFT 길이: 두 신호 중 더 긴 길이의 2배 이상이 되도록 2의 거듭제곱으로 확장
    n = 1
    L = max(len(sig), len(ref))
    while n < L * 2:
        n *= 2  # zero‑padding → 주파수 해상도 향상

    # 주파수 영역으로 변환
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)

    # 상호스펙트럼 + PHAT 가중치(크기 정규화)
    R = SIG * np.conj(REF)
    denom = np.abs(R)
    denom[denom == 0] = 1e-12  # 0 나눗셈 방지
    R /= denom

    # 역 FFT로 상호상관계수(cc) 계산 + 보간(interp 배)
    cc = np.fft.irfft(R, n=n * interp)
    max_shift = (n * interp) // 2
    # 원형 시프트를 중앙 기준으로 재배열(음/양 지연 모두 포함)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    # 피크 위치 → 샘플 시프트 → 시간(초)
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)

    # (선택) 물리 제한 적용: 최대 지연 초과 시 클리핑
    if max_tau is not None:
        max_shift_allowed = int(interp * fs * max_tau)
        if abs(shift) > max_shift_allowed:
            shift = np.clip(shift, -max_shift_allowed, max_shift_allowed)
            tau = shift / float(interp * fs)
    return tau, cc


# -----------------------------------------------------------------------------
# TDOA 비용 맵 계산 + 2D 격자 탐색 기반 위치 추정
# -----------------------------------------------------------------------------

def compute_tdoa_map(
    signals: List[np.ndarray],
    fs: int,
    mic_ids: List[str],
    mic_xy: Dict[str, Tuple[float, float]],
    ref_id: str,
    grid: Tuple[float, float, float, float, float],
    c: float = 343.0,
    max_pair_dist_m: float | None = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[float, float]]:
    """관측 신호들과 마이크 좌표를 바탕으로 2D 비용 맵과 최솟값 좌표를 계산합니다.

    signals : list[np.ndarray]
        마이크별 시계열 (모두 동일 길이/샘플레이트라고 가정. 길이는 최소값에 맞춰 컷)
    fs : int
        샘플레이트(Hz)
    mic_ids : list[str]
        signals의 순서와 동일한 마이크 ID 리스트
    mic_xy : dict[id -> (x,y)]
        각 마이크의 좌표(m)
    ref_id : str
        기준 마이크 ID (모든 지연은 ref 대비 계산)
    grid : (xmin, xmax, ymin, ymax, step)
        2D 격자 정의(단위 m)
    c : float
        음속(m/s). 기본 343
    max_pair_dist_m : float | None
        GCC‑PHAT에서 허용할 최대 마이크 간 거리(→ 최대 지연 한계 설정)

    Returns
    -------
    heat : np.ndarray [Ny, Nx]
        지점별 총 비용(작을수록 관측치와 부합)
    (X, Y) : meshgrid
        격자 좌표 그리드
    (x_hat, y_hat) : float, float
        최소 비용 지점(추정 위치)
    """
    # --- 0) 인덱스 매핑 및 ref 위치 확인 -------------------------------
    id2idx = {mid: i for i, mid in enumerate(mic_ids)}
    assert ref_id in id2idx, f"ref_id({ref_id}) not in mic_ids"
    ref_idx = id2idx[ref_id]

    # --- 1) 전처리: 신호 길이 정렬 + DC 제거 ----------------------------
    L = min(len(s) for s in signals)  # 가장 짧은 길이에 맞춰 컷팅
    sig = [zero_mean(s[:L]) for s in signals]

    # --- 2) 관측 지연(obs_tau) 계산: 각 마이크 vs ref --------------------
    obs_tau: Dict[str, float] = {}
    max_tau = None
    if max_pair_dist_m is not None:
        max_tau = max_pair_dist_m / c  # 거리 한계를 시간 한계로 변환

    for i, mid in enumerate(mic_ids):
        if i == ref_idx:
            obs_tau[mid] = 0.0
            continue
        tau, _ = gcc_phat(sig[i], sig[ref_idx], fs=fs, max_tau=max_tau, interp=16)
        obs_tau[mid] = float(tau)

    # --- 3) 2D 격자에서 비용 Σ|τ_pred − τ_obs| 계산 ----------------------
    xmin, xmax, ymin, ymax, step = grid
    xg = np.arange(xmin, xmax + 1e-9, step, dtype=np.float32)
    yg = np.arange(ymin, ymax + 1e-9, step, dtype=np.float32)
    X, Y = np.meshgrid(xg, yg)
    heat = np.zeros_like(X, dtype=np.float32)

    xr, yr = mic_xy[ref_id]
    for mid in mic_ids:
        if mid == ref_id:
            continue
        xi, yi = mic_xy[mid]
        # 후보점 P(x,y)에서 각 마이크까지의 거리 차이 → 예상 지연
        # tau_pred(x,y) = (|P−Mi| − |P−Mr|) / c
        di = np.sqrt((X - xi) ** 2 + (Y - yi) ** 2)
        dr = np.sqrt((X - xr) ** 2 + (Y - yr) ** 2)
        tau_pred = (di - dr) / c
        tau_obs = obs_tau[mid]
        # 관측과 예상의 차이 절댓값을 누적 (L1 비용)
        heat += np.abs(tau_pred - tau_obs).astype(np.float32)

    # --- 4) 최소 비용 지점 추출 -----------------------------------------
    idx = np.unravel_index(np.argmin(heat), heat.shape)
    x_hat, y_hat = float(X[idx]), float(Y[idx])
    return heat, (X, Y), (x_hat, y_hat)


# -----------------------------------------------------------------------------
# 파일 로딩/매핑 도우미
# -----------------------------------------------------------------------------

@dataclass
class MicSpec:
    id: str
    xy: Tuple[float, float]
    file: str


def expand_files(patterns: List[str]) -> List[str]:
    """글롭 패턴 목록을 확장하여 정렬된 고유 파일 리스트 반환.
    - patterns: ["Sensor-01_*.wav", ...]
    - 반환: 정렬된 파일 경로 목록
    - 일치 항목이 없으면 예외 발생
    """
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No files matched: {patterns}")
    return files


def parse_mic_args(mic_args: List[str]) -> Dict[str, Tuple[float, float]]:
    """"ID:x,y" 형태를 파싱하여 {ID: (x,y)} 딕셔너리로 반환.
    - 예: "Sensor-01:0,0" → {"Sensor-01": (0.0, 0.0)}
    - 파싱 실패 시 ValueError
    """
    out: Dict[str, Tuple[float, float]] = {}
    for s in mic_args:
        try:
            left, right = s.split(":")
            x_str, y_str = right.split(",")
            out[left.strip()] = (float(x_str), float(y_str))
        except Exception as e:
            raise ValueError(f"--mic 인자 파싱 실패: {s} (예: Sensor-01:0,0)") from e
    return out


# -----------------------------------------------------------------------------
# 시각화(옵션)
# -----------------------------------------------------------------------------

def plot_heatmap(
    heat: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    mic_xy: Dict[str, Tuple[float, float]],
    x_hat: float,
    y_hat: float,
    title: str = "TDOA 2D Heatmap",
) -> None:
    """matplotlib으로 heatmap + 마이크 위치 + 추정점 시각화."""
    import matplotlib.pyplot as plt

    # imshow는 기본적으로 y축이 아래로 증가하므로 origin="lower"로 설정하여
    # meshgrid 좌표계와 일치시키고, extent로 실제 축 범위를 지정
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    plt.figure()
    plt.imshow(heat, origin="lower", extent=extent, aspect="equal")
    plt.colorbar(label="Sum |τ_pred − τ_obs| (s)")

    # 마이크 위치 표시(삼각형) + 라벨
    for mid, (mx, my) in mic_xy.items():
        plt.scatter([mx], [my], marker="^", s=80)
        plt.text(mx + 0.02, my + 0.02, mid, fontsize=9)

    # 추정 위치 표시(X 마크)
    plt.scatter([x_hat], [y_hat], marker="x", s=100)

    plt.title(title)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 메인/CLI
# -----------------------------------------------------------------------------

def main():
    """명령행 인터페이스 진입점.
    - 파일 패턴을 확장하고, 마이크 좌표/기준/격자 설정을 읽어 실행합니다.
    """
    ap = argparse.ArgumentParser(description="2D TDOA (GCC‑PHAT) grid localization")

    # 입력 파일 관련 옵션
    g_files = ap.add_argument_group("입력 파일 옵션")
    g_files.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="글롭 패턴 또는 파일 경로 나열 (각 마이크 1개 파일이 이상적)",
    )

    # 마이크/기준 설정
    g_mic = ap.add_argument_group("마이크/기준 설정")
    g_mic.add_argument(
        "--mic",
        nargs="+",
        required=True,
        help='마이크 ID와 좌표. 예) --mic "Sensor-01:0,0" "Sensor-02:2,0" "Sensor-03:0,2"',
    )
    g_mic.add_argument("--ref", required=True, help="기준 마이크 ID")

    # 신호 전처리
    g_sig = ap.add_argument_group("신호 전처리")
    g_sig.add_argument("--fs", type=int, default=None, help="필요시 리샘플할 샘플레이트(Hz)")
    g_sig.add_argument(
        "--band",
        nargs=2,
        type=float,
        default=None,
        metavar=("F_LO", "F_HI"),
        help="Bandpass 필터(Hz) 적용. 예) --band 300 5000",
    )

    # 그리드 탐색
    g_grid = ap.add_argument_group("그리드 탐색")
    g_grid.add_argument(
        "--grid",
        nargs=5,
        type=float,
        required=True,
        metavar=("xmin", "xmax", "ymin", "ymax", "step"),
        help="탐색 범위/간격 (m). 예) --grid -1 3 -1 3 0.02",
    )

    # 기타
    g_misc = ap.add_argument_group("기타")
    g_misc.add_argument("--c", type=float, default=343.0, help="음속 m/s (기본 343)")
    g_misc.add_argument(
        "--maxdist",
        type=float,
        default=None,
        help="GCC‑PHAT 허용 최대 마이크 간 거리(m). 설정 시 지연 한계 설정",
    )
    g_misc.add_argument("--show", action="store_true", help="heatmap 시각화")

    args = ap.parse_args()

    # 1) 파일 패턴 확장
    files = expand_files(args.files)

    # 2) 마이크 좌표 파싱
    mic_xy = parse_mic_args(args.mic)

    # 3) 파일 ↔ 마이크 매핑 및 로딩
    #    - 파일명에 마이크 ID 문자열이 포함되어 있다고 가정하고 자동 매칭
    mic_ids: List[str] = []
    sigs: List[np.ndarray] = []
    fs_ref: int | None = None

    for mid in mic_xy.keys():
        # files 중 해당 mid가 포함된 파일들 후보
        cand = [f for f in files if mid in os.path.basename(f)]
        if not cand:
            raise FileNotFoundError(f"파일 목록에서 '{mid}'에 해당하는 파일을 찾지 못했습니다.")
        # 단순 정책: 사전순/시간순 가정으로 가장 마지막(보통 최신)을 선택
        path = sorted(cand)[-1]

        # WAV 로드(필요 시 리샘플)
        x, fs = load_wav_mono(path, target_fs=args.fs)

        # (선택) 대역통과 필터
        if args.band:
            x = bandpass_iir(x, fs, args.band[0], args.band[1])

        mic_ids.append(mid)
        sigs.append(x)
        fs_ref = fs if fs_ref is None else fs_ref
        if fs != fs_ref:
            # 서로 다른 샘플레이트가 들어오면 계산 불가 → --fs로 강제 정렬 필요
            raise RuntimeError("모든 신호의 샘플레이트가 일치해야 합니다. (--fs 옵션으로 강제 리샘플)")

    # 4) TDOA 비용 맵 + 추정 위치 계산
    heat, (X, Y), (x_hat, y_hat) = compute_tdoa_map(
        signals=sigs,
        fs=fs_ref,  # type: ignore[arg-type]
        mic_ids=mic_ids,
        mic_xy=mic_xy,
        ref_id=args.ref,
        grid=tuple(args.grid),  # type: ignore[arg-type]
        c=args.c,
        max_pair_dist_m=args.maxdist,
    )

    # 5) 결과 출력 + (옵션) 시각화
    print("[TDOA] Estimated position: x=%.3f m, y=%.3f m" % (x_hat, y_hat))
    if args.show:
        plot_heatmap(heat, X, Y, mic_xy, x_hat, y_hat, title="TDOA 2D Heatmap")


if __name__ == "__main__":
    main()
