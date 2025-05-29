# Flask 기반의 웹 서버 실행 파일
# 웹페이지 라우팅과 데이터베이스 연동을 통해 알림 정보를 사용자에게 전달

import os

# Flask 관련 모듈 불러오기
from flask import Flask, render_template, jsonify
# DB에서 알림 정보를 가져오는 함수
from web.utils.db_helper import get_alerts

# Flask 앱 생성 (템플릿과 정적 파일 경로 지정)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)
# 루트 페이지 라우팅 (메인 웹페이지 렌더링)
@app.route('/')
def index():
    return render_template('index.html')

# 알림 데이터 API 라우팅 (JSON 형태로 반환)
@app.route('/alerts')
def alerts():
    return jsonify(get_alerts())

# 서버 실행 설정 (모든 네트워크 인터페이스에서 접속 가능, 디버그 모드 ON)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)