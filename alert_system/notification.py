import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
import time
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')

def save_alert_to_db(alert, db_path="./DB/alerts.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            room_id TEXT NOT NULL,
            category TEXT NOT NULL,
            decibel REAL,
            priority INTEGER,
            original_type TEXT,
            processed INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        INSERT INTO alerts (created_at, room_id, category, decibel, priority, original_type)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        alert['time'].strftime('%Y-%m-%d %H:%M:%S'),
        alert['mic'],
        alert['type'],
        alert['decibel'],
        alert['priority'],
        alert['original_type']
    ))

    conn.commit()
    conn.close()

# 중요도 테이블 정의
priority_table = {
    'fire alarm': 1,
    'gas alarm': 1,
    'baby calling': 1,
    'phone ringing': 3,
    'home appliance(working)': 3,
    'doorbell': 3,
    'knocking': 3,
    'human voice misc': 4,
    'door movement': 4,
    'musical instrument(electric sound)': 4,
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

def load_recent_events(db_path, last_processed_time):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 데이터베이스에서 필요한 컬럼만 가져오기
    cursor.execute("SELECT created_at, decibel, room_id, category FROM inference_results")
    rows = cursor.fetchall()
    logging.info(f"{len(rows)}개의 데이터를 데이터베이스에서 불러왔습니다.")
    conn.close()

    now = datetime.now()
    logging.debug(f"Current time (now): {now}, Last processed time: {last_processed_time}")
    data = []
    seen = set()

    for row in rows:
        #logging.info(f"Raw DB row: {row}")
        time_str, decibel, mic, sound_type = row
        try:
            time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        except Exception as e:
            logging.error(f"Error parsing date '{time_str}': {e}")
            continue

        # 최근 처리된 이후의 데이터만, 30초 이내
        if last_processed_time < time_obj <= now and now - time_obj <= timedelta(seconds=30):
            logging.debug(f"Event passed time filter: {time_obj}, {decibel}, {mic}, {sound_type}")
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
    return data, now

def group_events_by_time(events):
    grouped = defaultdict(list)
    for item in events:
        grouped[item['time']].append(item)
    logging.info(f"{len(grouped)}개의 시간 그룹으로 데이터를 묶었습니다.")
    return grouped

def select_event_from_group(items):
    if len(items) == 1:
        selected = items[0]
    elif all(x['type'] == items[0]['type'] for x in items):
        selected = max(items, key=lambda x: x['decibel'])
    else:
        items.sort(key=lambda x: x['priority'])
        top3 = items[:3]
        selected = top3[0]
    return selected


def process_data(db_path, last_processed_time, cooldown_tracker):
    data, now = load_recent_events(db_path, last_processed_time)

    # Step 1: Group by room (mic)
    room_groups = defaultdict(list)
    for item in data:
        room_groups[item['mic']].append(item)

    alerts_to_send = []

    for room, events in room_groups.items():
        top_event = sorted(events, key=lambda x: x['priority'])[0]
        key = (top_event['mic'], top_event['type'])
        # cooldown 확인 (10 sec)
        if key not in cooldown_tracker or (now - cooldown_tracker[key]).total_seconds() > 10:
            alerts_to_send.append(top_event)
            cooldown_tracker[key] = now

    for alert in alerts_to_send:
        logging.info(f"{alert['type']} 소리가 {alert['mic']}에서 발생했습니다.")
        logging.info(f"선택된 소리: {alert['type']}, 마이크: {alert['mic']}, 우선순위: {alert['priority']}")
        save_alert_to_db(alert)

    return now  # 최근 처리 시간 반환

def start_alert_checker(db_path="./DB/inference_results.db"):
    last_time = datetime.now() - timedelta(seconds=5)
    cooldown_tracker = {}
    iteration_count = 0  # Add this line

    try:
        while True:
            last_time = process_data(db_path, last_time, cooldown_tracker)
            time.sleep(1)
            iteration_count += 1

            if iteration_count % 60 == 0:  # every 60 seconds
                # inference_results 유지
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM inference_results
                    WHERE id NOT IN (
                        SELECT id FROM inference_results
                        ORDER BY created_at DESC
                        LIMIT 1000
                    )
                """)
                conn.commit()
                conn.close()

                # alerts 유지
                alerts_conn = sqlite3.connect("./DB/alerts.db")
                alerts_cursor = alerts_conn.cursor()
                alerts_cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        room_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        decibel REAL,
                        priority INTEGER,
                        original_type TEXT,
                        processed INTEGER DEFAULT 0
                    )
                """)
                alerts_cursor.execute("""
                    DELETE FROM alerts
                    WHERE id NOT IN (
                        SELECT id FROM alerts
                        ORDER BY created_at DESC
                        LIMIT 300
                    )
                """)
                alerts_conn.commit()
                alerts_conn.close()
    except KeyboardInterrupt:
        logging.info("중지되었습니다.")
