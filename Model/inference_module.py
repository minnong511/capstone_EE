import os
import time 
import sys
import torch
import sqlite3
import logging
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
try:
    import soundfile as sf
    _HAVE_SF = True
except Exception:
    _HAVE_SF = False
    import wave
MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
DB_PATH = PROJECT_ROOT / "DB" / "inference_results.db"

project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.append(project_root_str)

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
conn = sqlite3.connect(str(DB_PATH), check_same_thread=False) ## check_thread에 대한 내용 필요 
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


# Helper: compute dBFS from audio file path
def _compute_dbfs(file_path: str) -> int:
    """Compute RMS dBFS from audio file. Returns integer dBFS for DB storage."""
    try:
        if _HAVE_SF:
            data, fs = sf.read(file_path, always_2d=True)
            x = data.mean(axis=1).astype('float32')
        else:
            with wave.open(file_path, 'rb') as w:
                fs = w.getframerate()
                n = w.getnframes()
                raw = w.readframes(n)
                ch = w.getnchannels()
            import numpy as _np
            x = _np.frombuffer(raw, dtype=_np.int16).astype('float32') / 32768.0
            if ch > 1:
                x = x.reshape(-1, ch).mean(axis=1)
        # zero-mean
        if x.size:
            x = x - float(x.mean())
        rms = float(np.sqrt((x * x).mean() + 1e-12))
        dbfs = 20.0 * np.log10(rms + 1e-12)
        return int(round(dbfs))
    except Exception:
        # On failure, return a sentinel low value
        return -120


def start_inference_loop(real_time_folder, panns_model, classifier_model, label_dict, device):
    logging.info("실시간 추론 루프 시작됨.")

    last_cleanup_time = datetime.now()

    real_time_dir = Path(real_time_folder)

    while True:
        all_files = sorted(real_time_dir.glob("*.wav"))

        for wav_path in all_files:
            try:
                filename = wav_path.name
                # 1. 파일명 확장 변경해서 중복 방지
                #logging.info(f"파일 감지됨: {filename}")
                processing_path = wav_path.with_suffix(wav_path.suffix + ".processing")
                wav_path.rename(processing_path)
                
                # 2. 메타정보 추출 (새 규칙: 19700101-090015_Sensor-03_Room102.wav)
                #   parts = [ '19700101-090015', 'Sensor-03', 'Room102.wav' ]
                parts = filename.split("_")
                if len(parts) < 3:
                    raise ValueError(f"unexpected filename format: {filename}")
                date_time = parts[0]                 # '19700101-090015'
                sensor_id = parts[1]                 # 'Sensor-03' (현재 DB에는 저장하지 않음)
                room_id = parts[2].rsplit('.', 1)[0] # 'Room102'

                if '-' not in date_time:
                    raise ValueError(f"unexpected date-time token: {date_time}")
                date, time_str = date_time.split('-')

                # 파일 내용으로 dBFS 계산 (이전처럼 파일명에서 읽지 않음)
                decibel = _compute_dbfs(str(processing_path))

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

                processing_path.unlink(missing_ok=True)

            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

        time.sleep(5)
