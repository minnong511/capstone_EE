"""
Service run model (상세 주석 · TOC/입출력 포함)

[코드 목차]
  0) 개요 및 실행 모드 설명 (이 문단)
  1) 로깅 설정
  2) 모듈 임포트 경로 보정(sys.path)
  3) 외부 모듈에서 가져오는 실행 진입점 함수 임포트
  4) 모델 관련 클래스/유틸 임포트
  5) 실행 모드 플래그(RUN_ONCE)
  6) 감시 재시작 래퍼(_supervised_run)
  7) 모델/경로 초기화(디바이스, 라벨, 폴더, 모델 로드)
  8) 스레드 기동: BLE 수신기 / 추론 / 알림
  9) 메인 스레드에서 시각화 실행
 10) 정리와 운영 상 주의사항

[입출력 구조(예시)]
  입력(Input)
   - BLE 수신 파일: `./Input_data/real_input/*.wav` (ESP32→PC 전송된 2초 오디오 조각)
   - 모델 가중치: `./Model/pretrained/Cnn10.pth`, `./Model/classifier_model.pth`
   - 라벨 사전 소스: `./Dataset/Dataset` (클래스명↔인덱스 매핑 로딩)

  처리(Processing)
   - 스레드1(BLE): ESP32 장치와 BLE 통신 → 오디오 파일을 실시간 폴더에 저장
   - 스레드2(추론): 폴더 감시 → 새 wav 로드 → PANNs 임베딩(512d) → 전이분류기로 예측
   - 스레드3(알림): DB alerts 테이블 기록 및 이벤트 트리거
   - 메인(시각화): DB 기반 대시보드/모니터링

  출력(Output)
   - 분류/알림 결과: 로깅 + DB(alerts) 기록 (시각화 모듈이 사용)
   - 실시간 파일: `./Input_data/real_input` 안에 수신된 원본 wav 보관

[실행 모드]
  * 기본값(RUN_ONCE=0): 각 워커는 내부적으로 무한 루프를 돌고, 예외나 정상 반환으로 종료되면
    supervisor가 2초 후 자동 재시작합니다. (현장/상시 운영)
  * 테스트 모드(RUN_ONCE=1): 각 워커를 한 번만 실행 후 종료합니다. (단발 테스트)

[셸 예시]
    export RUN_ONCE=1   # 한 번만 실행 (테스트)
    unset RUN_ONCE      # 상시 실행(기본)

[운영 주의]
  * 본 파일은 오케스트레이션만 담당합니다. 실제 I/O와 로직은 각 모듈(node_BLE, inference_module,
    notification, dbvisual_module)에 구현되어야 합니다.
  * 예외가 발생해도 서비스가 계속 살아있도록 감시 재시작(_supervised_run)으로 보호합니다.
"""

import logging
# [왜 이 설정이 필요한가?]
#  - 운영 중 멀티스레드에서 어떤 워커가 남긴 로그인지 식별이 중요 → threadName 포함
#  - 현장 시각 동기화/장애 추적을 위해 초 단위 타임스탬프 필요 → datefmt 지정
#  - 기본 레벨 INFO: 정상 동작 흐름을 남기되, WARNING/ERROR는 즉시 눈에 띄도록
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import sys
import os
import time
import threading
import torch
import sqlite3  # (현재 파일에서는 직접 사용하지 않지만, 외부모듈과의 인터페이스를 고려해 남겨둠)

# [경로 보정 이유]
#  - 프로젝트 루트/서브패키지에서 실행 위치가 달라질 때 sibling 패키지 임포트 실패를 방지.
#  - __file__ 기준 절대경로를 sys.path에 넣어 import 오류를 예방합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -------- 외부 모듈에서 가져오는 실행 진입점 함수 ----------------------------
# 각 모듈은 내부에서 while True 루프 등을 돌며 '자체적으로 장기 실행'을 담당합니다.
from Model.inference_module import start_inference_loop           # 추론 루프 진입점
from alert_system.notification import start_alert_checker          # 알림 처리 루프 진입점
from data_visaualization.dbvisual_module import start_db_visualization  # 시각화(메인 스레드)

# node.node_BLE.run_listener_for_all(real_time_folder: str) → None
#  - 내부에서 무한루프를 돌며 BLE 연결/재시도/수신파일 저장을 처리(반환하지 않는 것이 정상)
#  - 테스트 모드에서는 한 번만 실행하도록 구현할 수도 있음
from node.node_wifi import run_listener_for_all                     # BLE 수신 루프 진입점

# -------- 모델 관련 유틸/클래스 임포트 ---------------------------------------
from Model.base_model_panns import (
    PANNsCNN10,            # 사전학습 CNN 백본 (512차원 임베딩 특징 추출)
    TransferClassifier,     # 전이학습용 분류기(입력 512 → 클래스 수)
    infer_audio,            # (다른 모듈에서 사용) 단일 오디오 추론 유틸
    get_device,             # cuda/cpu 자동 선택
    get_label_dict          # 라벨 사전 로드(클래스명 ↔ 인덱스)
)

# ============================ 실행 구성 ========================================
# 환경변수 RUN_ONCE가 '1'이면 단발 실행 모드, 아니면 상시감시 모드
RUN_ONCE: bool = os.environ.get("RUN_ONCE", "0") == "1"


def _supervised_run(fn, *args, **kwargs):
    """대상 함수(fn)를 감시(supervise)하면서 무한히 재시작하는 래퍼.

    동작 개요
    - 대상 함수가 예외로 크래시하든, 정상적으로 return 하든, 항상 2초 대기 후 재시작합니다.
    - 현장 운영에서 일시 네트워크 장애(예: BLE 끊김), 파일 핸들 오류 등으로
      함수가 종료되는 경우에도 자동 복구(재연결 효과)를 제공합니다.

    매개변수
    - fn: 장시간 실행을 기대하는 워커 함수(예: run_listener_for_all)
    - *args, **kwargs: 대상 함수에 그대로 전달할 인자

    참고
    - 워커 함수가 정상 반환하는 경우도 "비정상"으로 간주하여 재시작하는 설계입니다.
      (폴더 감시 루프, BLE 리스너 등은 장기 실행이 전제)
    """
    while True:
        try:
            logging.info(f"Starting worker: {getattr(fn, '__name__', str(fn))}")
            fn(*args, **kwargs)
            # 일반적으로 워커는 무한루프를 가져야 하므로, 여기까지 오면 비정상(정상반환)으로 간주
            logging.warning("Worker returned normally; restarting in 2s to keep service alive.")
        except Exception:
            # 예외 발생 시 스택트레이스를 포함하여 로깅한 뒤 2초 후 재시작
            logging.exception("Worker crashed; restarting in 2s...")
        time.sleep(2)


# ============================ 모델/경로 초기화 =================================
# 디바이스 및 라벨 사전 준비
device = get_device()  # CUDA 가용 시 cuda, 아니면 cpu

# [중요] 라벨/분류기 클래스 수 불일치에 대한 안내
#  - classifier_model.pth가 학습될 때의 클래스 수 != 현재 get_label_dict가 반환한 길이이면
#    로드 시 size mismatch 에러가 발생합니다.
#  - 해결: (1) label_dict 기준으로 분류기를 재학습/재저장 하거나, (2) 임시로 엄격 로드를 끄고
#          head만 재정의 후 미세튜닝. (엄격 로드 해제 예시는 inference 모듈 쪽에 넣는 것을 권장)
label_dict = get_label_dict(root_dir='./Dataset/Dataset')

# 실시간 입력 파일이 떨어질 디렉터리(상대경로). 필요 시 상위 폴더까지 생성
real_time_folder = "./Input_data/real_input"
os.makedirs(real_time_folder, exist_ok=True)

# 사전학습 PANNs 백본 로드 → 특징 512차원 추출용
panns_model = PANNsCNN10('./Model/pretrained/Cnn10.pth').to(device)

# 전이 분류기 로드(입력 512 → 클래스 수). 학습된 파라미터를 디바이스에 매핑해 로드
classifier_model = TransferClassifier(input_dim=512, num_classes=len(label_dict))
classifier_model.load_state_dict(torch.load('Model/classifier_model.pth', map_location=device))
classifier_model.to(device)
classifier_model.eval()  # 추론 모드 고정(드롭아웃/배치정규화 등의 학습 동작 비활성화)


# ============================ 스레드 기동 ======================================
# threading을 사용하여 다음 3개 워커를 병렬 기동:
#  1) BLE 리스너 → real_time_folder에 ESP32에서 전송된 녹음 파일 저장
#  2) 추론 루프  → real_time_folder에 새로 생기는 파일을 감지/로드하여 분류 수행
#  3) 알림 체크  → 추론 결과를 DB alerts 테이블에 기록하고 이벤트 처리
# 시각화는 메인 스레드에서 실행(웹 대시보드 또는 CLI 그래프 등)

if __name__ == '__main__':
    # --- 1) ESP32 BLE Listener -------------------------------------------------
    # RUN_ONCE=True(테스트): run_listener_for_all 자체를 한 번 호출하고 종료
    # RUN_ONCE=False(운영): _supervised_run 래퍼로 감시 재시작
    threading.Thread(
        target=(run_listener_for_all if RUN_ONCE else _supervised_run),  # RUN_ONCE=True → 직접 실행 / False → _supervised_run으로 감시 실행
        args=(() if RUN_ONCE else (run_listener_for_all,)),  # 위 target 선택에 맞춘 인자 전달 방식
        kwargs={"real_time_folder": real_time_folder},  # 수신 파일 저장 경로(상대경로)
        name="BLEListenerThread",
        daemon=True  # 메인 스레드 종료 시 함께 종료되도록 데몬 스레드로 설정
    ).start()

    # --- 2) Inference ----------------------------------------------------------
    # start_inference_loop 시그니처에 맞게 인자 전달
    #  - RUN_ONCE=True : target에 직접 start_inference_loop 지정, 인자 튜플은 (real_time_folder, 모델들, 라벨, 디바이스)
    #  - RUN_ONCE=False: _supervised_run에 진입점과 인자들을 넘겨 감시 실행
    threading.Thread(
        target=(start_inference_loop if RUN_ONCE else _supervised_run),  # RUN_ONCE=True → 직접 실행 / False → _supervised_run으로 감시 실행
        args=(
            (real_time_folder, panns_model, classifier_model, label_dict, device,)
            if RUN_ONCE
            else (start_inference_loop, real_time_folder, panns_model, classifier_model, label_dict, device)
        ),  # 위 target 선택에 맞춘 인자 전달 방식
        name="InferenceThread",
        daemon=True
    ).start()

    # --- 3) Alert Checker ------------------------------------------------------
    threading.Thread(
        target=(start_alert_checker if RUN_ONCE else _supervised_run),  # RUN_ONCE=True → 직접 실행 / False → _supervised_run으로 감시 실행
        args=(() if RUN_ONCE else (start_alert_checker,)),  # 위 target 선택에 맞춘 인자 전달 방식
        name="AlertThread",
        daemon=True
    ).start()

    # 현재 모드와 입력 폴더를 로그로 남겨 추후 문제 발생 시 재현/진단에 도움
    logging.info(
        "RUN_ONCE=%s — Workers will %s. BLE real_time_folder=%s",
        RUN_ONCE,
        ("run once" if RUN_ONCE else "be supervised & auto-restarted"),
        real_time_folder,
    )

    # --- 4) Visualization ------------------------------------------------------
    # 시각화 루틴은 메인 스레드에서 '동기 실행'되도록 설계되었습니다.
    # 이유: 일부 시각화/웹서버 프레임워크는 메인 스레드에서의 이벤트 루프를 기대하기 때문입니다.
    # (fastapi/streamlit/matplotlib 대화형 등 환경별 제약을 고려)
    start_db_visualization()

# [정리]
# - supervisor 래퍼는 대상 함수가 예외로 종료되거나 정상 반환되더라도 2초 후 재시작합니다.
# - BLE 장치 연결이 끊겨 함수가 종료되더라도 자동 재시도(재연결 효과)를 제공합니다.
# - 경로는 모두 상대경로를 사용하므로, 프로젝트 루트에서 실행해야 동일한 구조로 동작합니다.
#   (예: `python capstone_EE/main_no_socket.py`)
# - RUN_ONCE=0(기본)에서 스레드는 daemon=True 이므로 메인(시각화) 루프가 종료되면 함께 종료됩니다.
# - 프로젝트 루트에서 실행 권장: 상대경로 기반 자원(Model/, Dataset/, Input_data/)을 안정적으로 찾기 위함.
# - 장애 대응: 특정 워커가 반복적으로 크래시하면 해당 모듈 로그를 먼저 확인하고, _supervised_run의 2초 대기
#   시간을 늘리면(예: 5~10초) 장치 재연결 안정성이 개선되는 경우가 있습니다.