# 로그 확인 기능 

import logging

# 로그 기본 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import threading
from Model.inference_module import start_inference_loop
from node.simulate_node import start_node_simulation
from data_visaualization.dbvisual_module import start_db_visualization
from alert_system.notification import start_alert_checker

import os
import time 
import sys
import torch
import sqlite3
from torch.utils.data import DataLoader
from datetime import datetime, timedelta

from Model.base_model_panns import (
    PANNsCNN10,
    TransferClassifier,
    infer_audio,
    get_device, 
    get_label_dict
)


device = get_device()
label_dict = get_label_dict(root_dir='./Dataset/Dataset')
real_time_folder = "./Input_data/simulator_input"

panns_model = PANNsCNN10('./Model/pretrained/Cnn10.pth').to(device)
classifier_model = TransferClassifier(input_dim=512, num_classes=len(label_dict))
classifier_model.load_state_dict(torch.load('Model/classifier_model.pth', map_location=device))
classifier_model.to(device)
classifier_model.eval()


#  threading을 사용하면 병렬로 구현이 가능함. 
#  node simulator, start_inference_loop, start_alert_checker 가 동시에 돌아가는 기적
if __name__ == '__main__':
    threading.Thread(
    target=start_inference_loop,
    args=(real_time_folder, panns_model, classifier_model, label_dict, device),
    name="InferenceThread"
    ).start()

    threading.Thread(
        target=start_node_simulation,
        name="NodeSimThread"
    ).start()

    threading.Thread(
        target=start_alert_checker,
        name="AlertThread"
    ).start()

    # 실시간 시각화는 메인 쓰레드에서 실행해야 한다
    start_db_visualization()  