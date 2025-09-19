# Smart Home Assistance System for the Hearing Impaired

서울과학기술대학교 전기정보공학과 2025년 캡스톤디자인 프로젝트
---

## Quick Start

### macOS
1. **Install Homebrew** (필요 패키지 설치용)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. **Install git, tmux**
   ```bash
   brew install git tmux
   ```
3. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd capstone_EE
   chmod +x run_tmux.sh
   ```

### Windows (WSL)
1. **Enable WSL**
   ```powershell
   wsl --install
   ```
2. **Install git, tmux inside WSL**
   ```bash
   sudo apt update
   sudo apt install git tmux
   ```
3. **Clone and prepare the project**
   ```bash
   git clone <repository_url>
   cd capstone_EE
   chmod +x run_tmux.sh
   ```

### Python Environment
```bash
conda create -n capstone python=3.11
conda activate capstone
pip install -r requirements.txt
```
`requirements.txt`에 필요한 라이브러리가 정리되어 있습니다. 

### Run the System
```bash
./run_tmux.sh
```
성공 시 다음과 같은 메시지가 출력됩니다. 
```text
[OK] tmux 세션 2개 생성됨.
 - worker_socket 세션: tmux attach -t worker_socket
 - server 세션: tmux attach -t server
```
필요 시 `tmux attach -t <세션명>`으로 각각의 세션을 확인하고, 종료는 `tmux kill-session`을 사용합니다.

개별 실행이 필요하면 다음 명령을 사용하세요.
```bash
python server.py
python worker.py
```

---

## Project Structure

```plaintext
capstone_EE/
├─ server.py                # HTTP 수신 서버, 업로드된 WAV 저장
├─ worker.py                # 사운드 추론 및 알림 필터링
├─ worker_socket.py         # 웹소켓 전송 기능을 포함한 워커
├─ web/
│  └─ websocket_server.py   # 실시간 알림 웹소켓 서버
├─ Model/
│  ├─ base_model_panns.py   # 백본 네트워크 
│  ├─ inference_module.py   # 데이터 추론용 모듈 
│  ├─ models.py
│  ├─ pytorch_utils.py
│  └─ training.py
├─ alert_system/
│  └─ notification.py       # 중복 알림 필터링 및 전송 로직
├─ node/
│  ├─ node_wifi.py          # Flask 업로드 서버 및 CLI 유틸리티
│  └─ wifi.ino               # ESP32 펌웨어
├─ DB/
│  ├─ inference_results.db
│  ├─ dataset_check.py
│  ├─ dataset_vis.py
│  ├─ insert.py
│  ├─ query.py
│  └─ sql_check.py
├─ Input_data/
│  ├─ real_input/           # 실시간 업로드 저장 경로
│  └─ simulator_input/
├─ data_visaualization/
│  └─ dbvisual_module.py
└─ run_tmux.sh
```

---

## System Overview

센서 노드는 각 방에 설치되며, 센서들이 실시간으로 소리를 수집합니다. 다만 모든 소리가 녹음되는 것은 아니며,사생활보호를 위해 임계치를 초과한 소리만이 녹음됩니다.  
임계치를 초과한 소리가 감지되면 서버로 전송되고, 서버는 딥러닝 모델(PANNs 백본 기반)로 사건을 분류합니다. 

이후 알림 관리 알고리즘이 중복 및 불필요한 알림을 제거하고, 사용자에게 이벤트 유형과 위치를 전달합니다. 시각화 도구와 모바일 앱이 동일한 DB(`DB/inference_results.db`)를 공유하여 최신 결과를 확인합니다.

### Workflow
1. 센서 노드가 주변 소음을 기록하고 임계치 기반으로 이벤트를 추출
2. 녹음된 WAV 파일을 서버로 업로드 (`server.py` → `Input_data/real_input/`)
3. `worker.py`가 새 파일을 감지하여 추론 수행 및 결과를 DB에 저장
4. `alert_system/notification.py`가 우선순위와 중복 여부를 판별
5. 웹소켓 서버(`web/websocket_server.py`)와 모바일 앱이 알림을 수신
6. 데이터 시각화 모듈이 최신 결과를 표시하고 로그를 확인


---

## Project Milestones

### 1. System Design
- Hardware Architecture 및 Configuration 설정 
    - ESP32 
    - INMP441 MEMS MIC
- 중앙 서버 구성, 통신 프로토콜 정의
    - Main server : Raspberry Pi5 8GB 
    - ESP32 - Server
        - Python Flask 
    - Server - APP
        - WebSocket


### 2. Sound Classification
- 데이터셋 구축 및 전처리
- PANNs 기반 전이학습과 분류기 설계
- 클래스 정의 및 라벨 매핑, 임베딩 추출 파이프라인 구축

### 3. Alert Management
- 소리 클래스별 알림 우선순위 설정
- 진동/시각화/로그 기반의 맞춤 알림 제공

### 4. Implementation
- ESP32 ↔ 서버 통신 통합
- 데이터 플로우 및 시스템 워크플로 검증
- 구현 이미지 및 세부 설명 정리

![DB Visualization](Image/db_visualization.png)
추론 결과가 SQLite DB에 반영된 화면 예시입니다.

### 5. Android App
- 서버 이벤트를 HTTP/WebSocket으로 수신하여 워커 알림과 동기화 (완료)
- 재전송 전략 및 백엔드 연계 계층 구현 (완료)
- Room DB 및 WorkManager 기반 로컬 로그 관리 (완료)
- 접근성 중심 UI, 이벤트 타임라인, 방별 필터 제공 (완료)
- 우선순위별 푸시/진동 채널 지원, 사용자 선호도 설정 (완료)
- UI 테스트 및 QA 체크리스트 정비 (진행 중)

![Connection Log](Image/connection_log.png)
실시간 업로드가 접속 로그에 기록되는 모습을 확인할 수 있습니다.

### 6. Evaluation
- 모의 테스트 정확도 98% 달성
- 실환경 테스트 진행 중, 사용자 피드백 수집 중

---


