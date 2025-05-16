import socket
import threading
import os
import time

# 1. 고정 IP 설정
NODE_IPS = {
    "room1": "192.168.0.101",
    "room2": "192.168.0.102",
    "room3": "192.168.0.103"
}
PORT = 5000
RECV_PORT = 6000
SAVE_DIR = "./Input_data/real_input"

os.makedirs(SAVE_DIR, exist_ok=True)

# 2. 연결 확인 함수
def check_connection(ip, port=PORT):
    try:
        with socket.create_connection((ip, port), timeout=2):
            return True
    except:
        return False

# 3. 트리거 및 파일 수신 루프
def listen_for_trigger(ip, port=RECV_PORT):
    while True:
        try:
            with socket.create_connection((ip, port)) as conn:
                filename = conn.recv(1024).decode()
                filepath = os.path.join(SAVE_DIR, filename)
                with open(filepath, 'wb') as f:
                    while True:
                        data = conn.recv(4096)
                        if not data:
                            break
                        f.write(data)
                print(f"[{ip}] Received: {filename}")
        except Exception as e:
            print(f"[{ip}] Error: {e}")
        time.sleep(1)  

def get_connected_nodes():
    connected = []
    for room, ip in NODE_IPS.items():
        if check_connection(ip):
            print(f"{room} ({ip}) 연결됨")
            connected.append(ip)
        else:
            print(f"{room} ({ip}) 연결 안 됨")
    return connected

def run_listener_for_all():
    connected_ips = get_connected_nodes()
    for ip in connected_ips:
        listen_for_trigger(ip)