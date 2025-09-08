# DB에서 알람 데이터 읽어서 실시간 heatmap 시각화

import sqlite3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
import logging
from Model.base_model_panns import (
    get_label_dict
)


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 라벨 딕셔너리를 불러와서 카테고리 목록을 정렬
label_dict = get_label_dict(root_dir='./Dataset/Dataset')
categories = sorted(label_dict.keys())

# 전체 room_id 목록을 저장하는 전역 변수
all_room_ids = []

# DB에서 room_id 목록을 불러오는 함수
def fetch_room_data():
    """
    DB에서 최근 100개의 알람 데이터에서 room_id를 불러와서 중복 없이 정렬된 목록을 반환합니다.
    """
    global all_room_ids
    if not all_room_ids:
        conn = sqlite3.connect('./DB/alerts.db')
        cursor = conn.cursor()
        cursor.execute("SELECT room_id FROM alerts ORDER BY id DESC LIMIT 100")
        results = cursor.fetchall()
        conn.close()
        all_room_ids = sorted(set(r[0] for r in results))
    return all_room_ids

import numpy as np
from datetime import datetime, timedelta

# 30초 이내의 알람 데이터를 DB에서 쿼리하여 heatmap 데이터를 생성하고 시각화하는 함수
def update_heatmap(i, ax):
    """
    DB에서 최근 30초 이내 생성된 알람 데이터를 조회하여,
    room_id와 카테고리별로 발생 빈도를 집계한 heatmap을 그립니다.
    """
    ax.clear()
    conn = sqlite3.connect('./DB/alerts.db')
    cursor = conn.cursor()
    threshold_time = (datetime.now() - timedelta(seconds=30)).strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("SELECT room_id, category, created_at FROM alerts WHERE created_at >= ?", (threshold_time,))
    results = cursor.fetchall()
    conn.close()

    if not results:
        ax.set_title("No recent alerts (last 30 seconds)")
        return

    room_ids = fetch_room_data()

    room_index = {room: idx for idx, room in enumerate(room_ids)}
    category_index = {cat: idx for idx, cat in enumerate(categories)}

    heatmap_data = np.zeros((len(room_ids), len(categories)))

    for room, category, _ in results:
        r = room_index[room]
        c = category_index[category]
        heatmap_data[r, c] += 1

    ax.imshow(heatmap_data, cmap='YlOrRd', interpolation='nearest', aspect='equal')
    ax.set_title("Real-time Alert Grid (last 30 seconds)")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(room_ids)))
    ax.set_xticklabels(categories, rotation=90, ha='center', fontsize=8)
    ax.set_yticklabels(room_ids, fontsize=8)

    for r in range(len(room_ids)):
        for c in range(len(categories)):
            val = int(heatmap_data[r, c])
            ax.text(c, r, val, ha='center', va='center',
                    color='black' if val < heatmap_data.max() / 2 else 'white')

# matplotlib 애니메이션 루프를 설정하여 실시간 heatmap 시각화를 시작하는 함수
def start_db_visualization():
    """
    matplotlib의 FuncAnimation을 이용하여 update_heatmap 함수를 주기적으로 호출,
    실시간으로 DB 알람 데이터를 시각화하는 애니메이션을 실행합니다.
    """
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, update_heatmap, fargs=(ax,), interval=1000, cache_frame_data=False)
    plt.show()