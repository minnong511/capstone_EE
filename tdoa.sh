#!/usr/bin/env bash
set -Eeuo pipefail

# --- Project root (this script's directory) ---
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

# --- Conda activation (capstone) ---
if command -v conda >/dev/null 2>&1; then
  # Use conda's shell hook if available (works on macOS zsh/bash)
  eval "$(conda shell.bash hook 2>/dev/null || true)"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found in PATH. Ensure Miniconda/Anaconda is installed." >&2
  exit 1
fi

# Activate env for each process run
ACTIVATE_AND_RUN='conda activate capstone && cd "$SCRIPT_DIR" && '

# --- Commands ---
SERVER_CMD="${ACTIVATE_AND_RUN} python server_tdoa.py 2>&1 | tee -a \"$SCRIPT_DIR/logs/server_tdoa.log\""
WORKER_CMD="${ACTIVATE_AND_RUN} python worker_tdoa.py 2>&1 | tee -a \"$SCRIPT_DIR/logs/worker_tdoa.log\""

# --- Prefer tmux if available ---
if command -v tmux >/dev/null 2>&1; then
  # Create/replace tmux sessions
  SESSION1="tdoa_server"
  SESSION2="tdoa_worker"

  # Kill existing sessions with the same names (optional safety)
  tmux has-session -t "$SESSION1" 2>/dev/null && tmux kill-session -t "$SESSION1" || true
  tmux has-session -t "$SESSION2" 2>/dev/null && tmux kill-session -t "$SESSION2" || true

  tmux new-session -d -s "$SESSION1" "bash -lc '$SERVER_CMD'"
  tmux new-session -d -s "$SESSION2" "bash -lc '$WORKER_CMD'"

  echo "[OK] Launched in tmux sessions:"
  echo " - server: tmux attach -t $SESSION1"
  echo " - worker: tmux attach -t $SESSION2"
else
  # Fallback: background processes with nohup
  nohup bash -lc "$SERVER_CMD" >/dev/null 2>&1 & echo $! > logs/server_tdoa.pid
  nohup bash -lc "$WORKER_CMD" >/dev/null 2>&1 & echo $! > logs/worker_tdoa.pid
  echo "[OK] Launched in background. PIDs saved under logs/*.pid"
  echo " - Tail logs: tail -f logs/server_tdoa.log logs/worker_tdoa.log"
fi