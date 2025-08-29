1.	tmux 세션 2개를 각각 별도로 띄움
    •	하나는 worker 전용
    •	하나는 server 전용
2.	각 세션에서 conda activate capstone 실행 후
    •	worker 세션 → python worker.py
    •	server 세션 → python server.
    

# 스크립트 예시 
#!/usr/bin/env bash
# run_tmux.sh
# worker.py와 server.py를 각각 별도 tmux 세션으로 실행

PROJECT_DIR="/Users/minnong511/code_repository/capstone_EE"
ENV_ACTIVATE="conda activate capstone"

# 1. worker 세션 실행
tmux new-session -d -s worker -n worker
tmux send-keys -t worker "cd $PROJECT_DIR" C-m
tmux send-keys -t worker "$ENV_ACTIVATE" C-m
tmux send-keys -t worker "python worker.py" C-m

# 2. server 세션 실행
tmux new-session -d -s server -n server
tmux send-keys -t server "cd $PROJECT_DIR" C-m
tmux send-keys -t server "$ENV_ACTIVATE" C-m
tmux send-keys -t server "python server.py" C-m

echo "[OK] tmux 세션 2개 생성됨."
echo " - worker 세션: tmux attach -t worker"
echo " - server 세션: tmux attach -t server"