# db 연동 로직 
import os
import sqlite3

def get_alerts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, 'DB', 'alerts.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT timestamp, room_id, category, priority FROM alerts ORDER BY timestamp DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows