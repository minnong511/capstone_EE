# main.py (예시)
import threading
from real_time_infer.real_time import start_inference
from alert_system.notification import start_alert_checker
from node_simulator.simulate_node import start_node_simulation

if __name__ == '__main__':
    threading.Thread(target=start_inference).start()
    threading.Thread(target=start_alert_checker).start()
    threading.Thread(target=start_node_simulation).start()