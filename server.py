# server.py
from node.node_wifi import run_listener_for_all

if __name__ == "__main__":
    # 같은 폴더(경로)로 맞춰야 워커가 파일을 읽음
    run_listener_for_all(real_time_folder="./Input_data/real_input",
                         host="0.0.0.0", port=5050)