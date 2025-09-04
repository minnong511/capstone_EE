# worker.py
import logging, time, torch
from Model.inference_module import start_inference_loop
from alert_system.notification import start_alert_checker
from Model.base_model_panns import (
    PANNsCNN10, TransferClassifier, get_device, get_label_dict
)
from data_visaualization.dbvisual_module import start_db_visualization
from node.tdoa import run_tdoa_loop
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = get_device()
label_dict = get_label_dict(root_dir='./Dataset/Dataset')

panns_model = PANNsCNN10('./Model/pretrained/Cnn10.pth').to(device)
classifier_model = TransferClassifier(input_dim=512, num_classes=len(label_dict))
classifier_model.load_state_dict(torch.load('Model/classifier_model.pth', map_location=device))
classifier_model.to(device)
classifier_model.eval()

real_time_folder = "./Input_data/real_input"  # ← server.py와 동일해야 함

if __name__ == "__main__":
    import threading

    threading.Thread(
        target=run_tdoa_loop,
        args=(Path(real_time_folder),),
        name="TDOAThread",
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
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[WORKER] Shutting down...")