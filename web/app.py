# 플라스크 서버 메인 파일 
from flask import Flask, render_template
from DB.query import get_alerts

app = Flask(__name__)

@app.route('/')
def index():
    alerts = get_alerts()
    return render_template('index.html', alerts=alerts)

if __name__ == '__main__':
    app.run(debug=True)