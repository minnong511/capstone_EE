# main.py (예시)
import threading
from Model.inference_module import run_batch_inference
from alert_system.notification import start_alert_checker
from node_simulator.simulate_node import start_node_simulation


#  threading을 사용하면 병렬로 구현이 가능함. 
#  node simulator, run_batch_inference, start_alert_checker 가 동시에 돌아가는 기적
if __name__ == '__main__':
    threading.Thread(target=run_batch_inference).start()
    threading.Thread(target=start_alert_checker).start()
    threading.Thread(target=start_node_simulation).start()