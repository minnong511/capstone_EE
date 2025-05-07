import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
import time
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')

# 중요도 테이블 정의
priority_table = {
    'fire alarm': 1,
    'gas alarm': 1,
    'baby calling': 1,
    'phone ringing': 3,
    'doorbell': 3,
    'knocking': 3,
    'human voice misc': 4,
    'door movement': 4,
    'musical instrument': 4,
    'snoring': 4,
    'home appliance': 4,
    'strong wind': 4,
    'tv sound': 4,
    'vacuum cleaner': 4,
}

# 'person calling'에 대한 데시벨 기준 우선순위 분류
def get_person_call_priority(decibel):
    if decibel < 30:
        return 'normal person calling', 4
    elif decibel < 50:
        return 'important person calling', 2
    else:
        return 'urgent person calling', 1

# 데이터 처리 함수
def process_data(db_path, last_processed_time):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 데이터베이스에서 필요한 컬럼만 가져오기
    cursor.execute("SELECT created_at, decibel, room_id, category FROM inference_results")
    rows = cursor.fetchall()
    conn.close()

    now = datetime.now()
    data = []
    seen = set()

    for row in rows:
        time_str, decibel, mic, sound_type = row
        time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

        # 최근 처리된 이후의 데이터만, 30초 이내
        if last_processed_time < time_obj <= now and now - time_obj <= timedelta(seconds=30):
            key = (sound_type, mic)
            if key not in seen:
                seen.add(key)

                # person calling에만 데시벨 우선순위 적용
                if sound_type == 'person calling':
                    label, priority = get_person_call_priority(decibel)
                    full_type = label
                else:
                    full_type = sound_type
                    priority = priority_table.get(sound_type, 99)

                data.append({
                    'time': time_obj,
                    'decibel': decibel,
                    'mic': mic,
                    'type': full_type,
                    'priority': priority,
                    'original_type': sound_type,
                })

    # 시간 단위로 그룹화
    grouped = defaultdict(list)
    for item in data:
        grouped[item['time']].append(item)

    # 그룹별 처리
    for time_group, items in grouped.items():
        if len(items) == 1:
            selected = items[0]
        elif all(x['type'] == items[0]['type'] for x in items):
            selected = max(items, key=lambda x: x['decibel'])
        else:
            items.sort(key=lambda x: x['priority'])
            top3 = items[:3]
            selected = top3[0]

        logging.info(f"{selected['type']} 소리가 {selected['mic']}에서 발생했습니다.")

    return now  # 최근 처리 시간 반환

def start_alert_checker(db_path="./DB/inference_results.db"):
    last_time = datetime.now() - timedelta(seconds=5)

    try:
        while True:
            last_time = process_data(db_path, last_time)
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("중지되었습니다.")
