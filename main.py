import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import threading
from Model.inference_module import start_inference_loop
# from alert_system.notification import start_alert_checker
from node.simulate_node import start_node_simulation


#  threading을 사용하면 병렬로 구현이 가능함. 
#  node simulator, start_inference_loop, start_alert_checker 가 동시에 돌아가는 기적
if __name__ == '__main__':
    threading.Thread(target=start_inference_loop).start()
    # threading.Thread(target=start_alert_checker).start()
    threading.Thread(target=start_node_simulation).start()