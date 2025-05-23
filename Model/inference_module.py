import os
import time 
import sys
import torch
import sqlite3
import logging
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.base_model_panns import (
    PANNsCNN10,
    TransferClassifier,
    infer_audio,
    get_device, 
    get_label_dict
)



# real_input으로 추후에 수정해야 하니깐 기억하자!
# 추론 실시간 처리 고려사항
# 파일을 실시간으로 읽는 게 제일 중요
#   고려사항들
#   만약 추론 처리 시간이 5초보다 오래 걸려서, 다음 루프에서 동일 파일이 또 처리되는 경우는 어떻게 방지?
#   이 경우에는 파일을 읽는 방식이 맨 뒷자리 .wav 니까 -> 맨 뒷자리 확장자 바꿔서 못 읽게 



# SQLite 테이블 생서 
# -> 테이블을 저장해야 함 .
# 한 번만 저장하면 될 듯? 
# 추론 결과 저장 
    # 소리 분류를 계속 저장해야 함. 
    # 즉 DB에 insert 되어야 함 .

# 전역 DB 설정 (클래스 내부에서 처리)
conn = sqlite3.connect("./DB/inference_results.db", check_same_thread=False) ## check_thread에 대한 내용 필요 
cursor = conn.cursor()

# 테이블 생성 (초기 1회)
cursor.execute("""
CREATE TABLE IF NOT EXISTS inference_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id TEXT,
    date TEXT,
    time TEXT,
    category TEXT,
    decibel INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()


def start_inference_loop(real_time_folder, panns_model, classifier_model, label_dict, device):
    logging.info("실시간 추론 루프 시작됨.")

    last_cleanup_time = datetime.now()

    while True:
        all_files = [f for f in os.listdir(real_time_folder) if f.endswith(".wav")]

        for filename in all_files:
            try:
                # 1. 파일명 확장 변경해서 중복 방지
                #logging.info(f"파일 감지됨: {filename}")
                original_path = os.path.join(real_time_folder, filename)
                processing_path = original_path + ".processing"
                os.rename(original_path, processing_path)
                
                # 2. 메타정보 추출
                parts = filename.split("_")
                room_id = parts[0]
                date = parts[1]
                time_str = parts[2].split(".")[0]
                decibel = parts[3].split(".")[0]

                # 3. 추론
                result = infer_audio(
                    file_path=processing_path,
                    room_id=room_id,
                    date=date,
                    time=time_str,
                    decibel=decibel, 
                    panns_model=panns_model,
                    classifier_model=classifier_model,
                    label_dict=label_dict,
                    device=device
                )
                #logging.info(f"추론 완료: {filename}")

                created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                cursor.execute("""
                    INSERT INTO inference_results (room_id, date, time, category, decibel, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (result["room_id"], result["date"], result["time"], result["category"], result["decibel"], created_at))

                conn.commit()
                #logging.info("DB 저장 완료")

                os.remove(processing_path)

            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

        time.sleep(5)