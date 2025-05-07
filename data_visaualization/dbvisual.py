import sqlite3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def fetch_data():
    # 데이터 수집
    # 최근 100개의 데이터를 수집 
    # DB에서 category 열에서 최근 100개 읽어옴 
    conn = sqlite3.connect('./DB/inference_results.db')
    cursor = conn.cursor()
    cursor.execute("SELECT category FROM inference_results ORDER BY id DESC LIMIT 100")
    results = cursor.fetchall()
    conn.close()
    # 리스트로 변환한다
    return [r[0] for r in results]

def update_chart(i, ax, bar_container):
    ax.clear()  # 기존 차트 초기화
    data = fetch_data()
    counter = Counter(data)

    if not counter:
        ax.set_title("No data to display")
        return

    # 카테고리 알파벳순 정렬
    categories, counts = zip(*sorted(counter.items()))

    bars = ax.bar(categories, counts)
    ax.set_ylabel("Count")
    ax.set_title("Real-time Sound Event Counts (last 100 events)")
    for bar, label in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, str(label), ha='center', va='bottom')

def start_db_visualization():
    # 2초 간격으로 시각화 창 다시 띄움
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, update_chart, fargs=(ax, None), interval=1000, cache_frame_data=False)
    plt.show()