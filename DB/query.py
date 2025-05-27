# 알림 조회용 함수 

import sqlite3

def get_alerts(db_path="DB/alerts.db"):
    conn = sqlite3.connect(db_path) # db 연동
    cursor = conn.cursor() # SQL 명령을 실행하기 위한 커서 객체 
    
    cursor.execute("SELECT timestamp, room_id, category, priority FROM alerts ORDER BY timestamp DESC")
    # 카테고리 열을 조회 , OREDR BY DESC -> 가장 최근의 알림부터 내림차순
    rows = cursor.fetchall()
    # 쿼리 결과를 가져와서 리스트 형태로 
    
    conn.close()
    # 데이터 베이스 연결 종료
    # return rows로 조회한 데이터 반환 
    return rows