# node simulator 
# 실제로 연결될 노드 시뮬레이터 
# train_Dataset에서 임의로 데이터를 불러오고, 제목을 roomid_date_time.wav로 변경
# 그리고 ./Input_data/simulator_input 으로 복사 예정
# --> 실행중에 삭제될 거니까 뭐 크게 걱정은 안해도 됨 

import os
import random
import shutil
import time
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def start_node_simulation(dataset_root='./Dataset/Dataset', output_dir='./Input_data/simulator_input', interval_sec=2):
    logging.info("Node 시뮬레이션 시작됨.")

    file_list = []
    for class_dir in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, class_dir)
        if os.path.isdir(class_path):
            wavs = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.wav')]
            file_list.extend(wavs)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    while time.time() - start_time < 100:  # 총 10초만 실행
        src_file = random.choice(file_list)
        # 랜덤으로 방 고르기 ㅋㅋ
        room_id = random.choice(['1', '2', '3'])

        now = datetime.now()
        date = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")

        decibel = random.randint(50, 100)  # 50~100dB 사이의 값 임의 생성

        new_filename = f"room{room_id}_{date}_{time_str}_{decibel}.wav"
        dst_path = os.path.join(output_dir, new_filename)

        shutil.copy(src_file, dst_path)
        logging.info(f"전송됨: {new_filename}")

        time.sleep(interval_sec)
