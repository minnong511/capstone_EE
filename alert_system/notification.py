import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta
import time
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s')

# 알림(alert) 정보를 데이터베이스에 저장하는 함수
def save_alert_to_db(alert, db_path="./DB/alerts.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # alerts 테이블이 없으면 생성
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

    # alert 정보를 테이블에 삽입
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

# 'person calling' 사운드에 대해 데시벨 기준으로 우선순위를 분류하는 함수
def get_person_call_priority(decibel):
    if decibel < 30:
        return 'normal person calling', 4
    elif decibel < 50:
        return 'important person calling', 2
    else:
        return 'urgent person calling', 1

# 최근 이벤트를 데이터베이스에서 로드하는 함수
# last_processed_time 이후의 데이터 중 30초 이내의 데이터만 필터링
def load_recent_events(db_path, last_processed_time):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # inference_results 테이블에서 필요한 컬럼만 선택해서 가져옴
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

        # 최근 처리된 이후의 데이터만, 30초 이내인 경우에만 처리
        if last_processed_time < time_obj <= now and now - time_obj <= timedelta(seconds=30):
            logging.debug(f"Event passed time filter: {time_obj}, {decibel}, {mic}, {sound_type}")
            key = (sound_type, mic)
            if key not in seen:
                seen.add(key)

                # person calling에 대해서만 데시벨 기반 우선순위 적용
                if sound_type == 'person calling':
                    label, priority = get_person_call_priority(decibel)
                    full_type = label
                else:
                    full_type = sound_type
                    priority = priority_table.get(sound_type, 99)  # 기본 우선순위는 99

                # 필터링된 이벤트 데이터를 리스트에 추가
                data.append({
                    'time': time_obj,
                    'decibel': decibel,
                    'mic': mic,
                    'type': full_type,
                    'priority': priority,
                    'original_type': sound_type,
                })
    return data, now

# 이벤트 리스트를 발생 시간별로 그룹화하는 함수
def group_events_by_time(events):
    grouped = defaultdict(list)
    for item in events:
        grouped[item['time']].append(item)
    logging.info(f"{len(grouped)}개의 시간 그룹으로 데이터를 묶었습니다.")
    return grouped

# 동일 시간 그룹 내에서 대표 이벤트를 선택하는 함수
def select_event_from_group(items):
    if len(items) == 1:
        selected = items[0]
    # 모두 같은 타입이면 데시벨이 가장 큰 것을 선택
    elif all(x['type'] == items[0]['type'] for x in items):
        selected = max(items, key=lambda x: x['decibel'])
    else:
        # 타입이 다르면 우선순위 기준으로 정렬 후 상위 3개 중 첫번째 선택
        items.sort(key=lambda x: x['priority'])
        top3 = items[:3]
        selected = top3[0]
    return selected

# 데이터 처리 및 알림 생성 함수
# 최근 처리 시간 이후의 데이터를 가져와서 방별로 우선순위가 가장 높은 이벤트를 선택
# 동일 이벤트에 대해 10초 쿨다운을 적용하여 중복 알림 방지
def process_data(db_path, last_processed_time, cooldown_tracker):
    data, now = load_recent_events(db_path, last_processed_time)

    # Step 1: 방(room)별로 이벤트 그룹핑
    room_groups = defaultdict(list)
    for item in data:
        room_groups[item['mic']].append(item)

    alerts_to_send = []

    # 각 방별로 우선순위가 가장 높은 이벤트를 선택하고 쿨다운 확인 후 알림 리스트에 추가
    for room, events in room_groups.items():
        top_event = sorted(events, key=lambda x: x['priority'])[0]
        key = (top_event['mic'], top_event['type'])
        # 쿨다운 확인 (10초)
        if key not in cooldown_tracker or (now - cooldown_tracker[key]).total_seconds() > 10:
            alerts_to_send.append(top_event)
            cooldown_tracker[key] = now

    # 선택된 알림을 로그에 출력하고 DB에 저장
    for alert in alerts_to_send:
        logging.info(f"{alert['type']} 소리가 {alert['mic']}에서 발생했습니다.")
        logging.info(f"선택된 소리: {alert['type']}, 마이크: {alert['mic']}, 우선순위: {alert['priority']}")
        save_alert_to_db(alert)

    return now  # 최근 처리 시간 반환

# 알림 체크를 주기적으로 수행하는 함수
# 1초마다 데이터를 처리하고, 60초마다 DB의 오래된 데이터 정리 수행
def start_alert_checker(db_path="./DB/inference_results.db"):
    last_time = datetime.now() - timedelta(seconds=5)
    cooldown_tracker = {}
    iteration_count = 0  # 실행 횟수 카운터 추가

    try:
        while True:
            last_time = process_data(db_path, last_time, cooldown_tracker)
            time.sleep(1)
            iteration_count += 1

            # 60초마다 inference_results와 alerts 테이블의 오래된 데이터 삭제
            if iteration_count % 60 == 0:  # every 60 seconds
                # inference_results 테이블 유지 작업
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

                # alerts 테이블 유지 작업
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
