# worker.py
import logging, time, torch, asyncio
from Model.inference_module import start_inference_loop
from alert_system.notification import start_alert_checker
from Model.base_model_panns import (
    PANNsCNN10,
    TransferClassifier,
    get_device,
    get_label_dict,
    DATASET_DIR,
    PRETRAINED_CHECKPOINT,
    CLASSIFIER_MODEL_PATH,
    safe_torch_load,
)
from data_visaualization.dbvisual_module import start_db_visualization
from web.websocket_server import start_websocket_server, start_db_event_publisher

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = get_device()
label_dict = get_label_dict(root_dir=str(DATASET_DIR))

panns_model = PANNsCNN10(PRETRAINED_CHECKPOINT).to(device)
classifier_model = TransferClassifier(input_dim=512, num_classes=len(label_dict))
classifier_state = safe_torch_load(CLASSIFIER_MODEL_PATH, map_location=device)
if isinstance(classifier_state, dict) and 'state_dict' in classifier_state:
    classifier_state = classifier_state['state_dict']
classifier_model.load_state_dict(classifier_state)
classifier_model.to(device)
classifier_model.eval()

real_time_folder = "./Input_data/real_input"  # ← server.py와 동일해야 함

def run_websocket():
    """
    웹소켓 서버를 독립 스레드에서 실행합니다.
    asyncio 이벤트 루프를 스레드마다 새로 생성하기 위해 asyncio.run()을 사용합니다.
    """
    try:
        asyncio.run(start_websocket_server(host="0.0.0.0", port=8765))
    except Exception as e:
        logging.exception("[WebSocket] 서버 실행 오류: %s", e)

"""
•	host="localhost"
→ 서버 머신의 로컬 네트워크 인터페이스에서만 접속을 허용하겠다는 뜻이에요.
→ 즉, 같은 머신에서만 ws://localhost:8765 로 접근 가능해집니다.
•	port=8765
→ 서버가 열리는 포트 번호예요. 기본 웹소켓 URL은 ws://localhost:8765.
"""

if __name__ == "__main__":
    import threading

    logging.info("[WORKER] Startup: inference, alert, db-visual, websocket 서버를 스레드로 실행합니다.")

    # 1) 웹소켓 서버
    threading.Thread(target=run_websocket, name="WebSocketThread", daemon=True).start()

    # 2) DB 이벤트 퍼블리셔 (개별 행을 순차 전송)
    threading.Thread(
        target=start_db_event_publisher,
        kwargs={"poll_ms": 500, "send_history": False, "batch_limit": 200},
        name="DBEventPublisher",
        daemon=True
    ).start()

    threading.Thread(
        target=start_inference_loop,
        args=(real_time_folder, panns_model, classifier_model, label_dict, device),
        name="InferenceThread",
        daemon=True
    ).start()

    threading.Thread(
        target=start_alert_checker,
        name="AlertThread",
        daemon=True
    ).start()

    start_db_visualization()


    # 메인 프로세스를 유지
    try:
        logging.info("[WORKER] All background services started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[WORKER] Shutting down...")
