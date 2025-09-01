# Repository Guidelines

## Project Structure & Module Organization
- `server.py`: Wi‑Fi HTTP receiver; saves WAV files to `Input_data/real_input`.
- `worker.py`: Runs inference + alert checker; reads from `Input_data/real_input`, writes results to `DB/inference_results.db`.
- `node/`: Device-side integration. `node_wifi.py` provides a Flask app/CLI for HTTP uploads.
- `Model/`: PANNs backbone, transfer classifier, and inference utilities; pretrained weights under `Model/pretrained/`.
- `alert_system/`: Notification logic.
- `DB/`: SQLite databases and small helpers.
- `Input_data/`: `real_input/` (live uploads) and `simulator_input/`.
- Other folders (`BLE_Test/`, `data_visaualization/`, `Beamforming/`) contain experiments/prototypes.

## Build, Test, and Development Commands
- Environment: `conda create -n capstone python=3.10` then `conda activate capstone`.
- Install deps: `pip install flask torch torchaudio pandas seaborn scikit-learn matplotlib`.
- Run both services (tmux): `bash ./run_tmux.sh`.
- Run individually: `python server.py` and `python worker.py`.
- Quick upload test: `curl -X POST --data-binary @sample.wav -H "X-Room-ID: Room101" -H "X-Mic-ID: Sensor01" http://localhost:5050/upload`.

## Coding Style & Naming Conventions
- Python 3.9+; follow PEP 8; 4‑space indentation.
- Filenames/functions: `snake_case`; classes: `PascalCase`.
- Place new modules under existing folders (`Model`, `node`, `alert_system`); prefer relative imports consistent with current code.
- Use type hints and short docstrings for new public functions.

## Testing Guidelines
- No formal unit test suite yet.
- Validate end‑to‑end:
  - Start `server.py` and `worker.py`.
  - Upload a WAV (see curl above).
  - Verify DB: `sqlite3 DB/inference_results.db "SELECT room_id, date, time, category FROM inference_results ORDER BY id DESC LIMIT 5;"`.
- If adding tests, create a `tests/` folder; name files `test_<module>.py`; prefer `pytest`.

## Commit & Pull Request Guidelines
- History uses short, imperative summaries (e.g., `wifi config`, `script added`). Keep messages concise and scoped.
- PRs should include: purpose, key changes, run steps, and screenshots/logs if UI/DB changes. Link issues and note any migration steps (DB/schema).

## Security & Configuration Tips
- Default HTTP port `5050`; restrict exposure or use a reverse proxy + HTTPS in production.
- Keep `server.py` and `worker.py` aligned on `Input_data/real_input`.
- Large uploads are saved atomically via `.part` → rename (handled in `node/node_wifi.py`).

