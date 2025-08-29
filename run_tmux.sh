#!/usr/bin/env bash
# run_tmux.sh
# worker.py와 server.py를 각각 별도 tmux 세션으로 실행

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
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

