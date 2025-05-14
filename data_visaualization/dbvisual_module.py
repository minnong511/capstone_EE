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

label_dict = get_label_dict(root_dir='./Dataset/Dataset')
categories = sorted(label_dict.keys())

def fetch_room_data():
    # 최근 100개의 알림에서 room_id만 추출
    conn = sqlite3.connect('./DB/alerts.db')
    cursor = conn.cursor()
    cursor.execute("SELECT room_id FROM alerts ORDER BY id DESC LIMIT 100")
    results = cursor.fetchall()
    conn.close()
    return [r[0] for r in results]

import numpy as np
from datetime import datetime, timedelta

def update_heatmap(i, ax):
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

    room_ids = sorted(set(r[0] for r in results))

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

def start_db_visualization():
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, update_heatmap, fargs=(ax,), interval=1000, cache_frame_data=False)
    plt.show()