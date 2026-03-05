# Webtoolbox

Webtoolbox is a modular FastAPI web app that hosts multiple tools.

Current tool:
- Audio -> Text Transcoder

## Architecture

- `src/webtoolbox/main.py`: FastAPI app, middleware, and home page
- `src/webtoolbox/config.py`: environment-driven settings
- `src/webtoolbox/logging_setup.py`: structured JSON logging with daily log files
- `src/webtoolbox/common/`: shared helpers
- `src/webtoolbox/tools/transcriber/`: tool module with routes, stores, progress tracking, and transcription service
- `templates/`: Jinja2 templates
- `static/`: CSS

The transcriber tool stores files under a single data root:
- Audio: `/data/webtoolbox/audio`
- Transcripts: `/data/webtoolbox/transcripts`

## Features

- Home page listing available tools
- Tool page with:
  - Existing audio and transcript file lists
  - Drag and drop upload for audio and text files
  - Audio and transcript delete operations
  - Transcript download operations
  - Model profile selection (`General` or `Special Estonian`)
  - Language selection (`Auto`, `Estonian`, `Russian`, `English`)
  - Background transcription jobs
  - Live progress updates over WebSocket
- Transcript output naming rule:
  - `meeting01.wav` -> `meeting01.txt`
- Structured log file for all key actions:
  - requests
  - uploads/downloads/rename/delete
  - transcription start/progress/finish/errors

## Logging

Logs are written to daily files:
- `logs/webtoolbox-YYYY-MM-DD.log`

Log lines are JSON with timestamp, level, logger, message, and metadata.

## Configuration

Environment variables:

- `WEBTOOLBOX_HOST` (default: `0.0.0.0`)
- `WEBTOOLBOX_PORT` (default: `8000`)
- `WEBTOOLBOX_DATA_DIR` (default: `/data/webtoolbox`)
- `WEBTOOLBOX_AUDIO_SUBDIR` (default: `audio`)
- `WEBTOOLBOX_TRANSCRIPT_SUBDIR` (default: `transcripts`)
- `WEBTOOLBOX_LOGS_DIR` (default: `logs`)
- `WEBTOOLBOX_MAX_UPLOAD_BYTES` (default: `209715200`)
- `WEBTOOLBOX_WHISPER_MODEL` (default: `small`)
- `WEBTOOLBOX_WHISPER_ESTONIAN_MODEL` (default: empty, required for `Special Estonian` profile)
- `WEBTOOLBOX_DEFAULT_MODEL_PROFILE` (default: `general`)
- `WEBTOOLBOX_DEFAULT_TRANSCRIBE_LANGUAGE` (default: `auto`)
- `WEBTOOLBOX_WHISPER_CPU_THREADS` (default: `0`, lets runtime choose)
- `WEBTOOLBOX_WHISPER_NUM_WORKERS` (default: `1`)
- `WEBTOOLBOX_WHISPER_BEAM_SIZE` (default: `1`, faster than beam size 5)

## Special Estonian model setup

`Special Estonian` uses a `faster-whisper` compatible model (CTranslate2 format).
You can provide either:

- A Hugging Face model id (for example: `org/model-name-ct2`)
- A local directory path where that model is stored

Example install flow in container:

```bash
cd /opt/webtoolbox
source .venv/bin/activate

# Optional helper for downloading model snapshots locally
pip install -U huggingface_hub

# Example: download a model snapshot to local storage
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="<HF_ESTONIAN_CT2_MODEL_ID>",
  local_dir="/opt/webtoolbox/models/estonian-ct2",
  local_dir_use_symlinks=False,
)
print("Model downloaded to /opt/webtoolbox/models/estonian-ct2")
PY
```

Then configure service environment (drop-in or service file):

```ini
Environment="WEBTOOLBOX_WHISPER_MODEL=small"
Environment="WEBTOOLBOX_WHISPER_ESTONIAN_MODEL=/opt/webtoolbox/models/estonian-ct2"
Environment="WEBTOOLBOX_DEFAULT_MODEL_PROFILE=general"
Environment="WEBTOOLBOX_DEFAULT_TRANSCRIBE_LANGUAGE=auto"
```

Reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart webtoolbox
```

Notes:

- If `WEBTOOLBOX_WHISPER_ESTONIAN_MODEL` is empty, `Special Estonian` is disabled in the UI.
- `Special Estonian` profile is expected to run through `faster-whisper`.

## Setup in Proxmox LXC (Linux)

1. Install system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg git
```

2. Clone project and create virtual environment:

```bash
git clone <your-repo-url> /opt/webtoolbox
cd /opt/webtoolbox
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Create data directories and permissions:

```bash
sudo mkdir -p /data/webtoolbox/audio /data/webtoolbox/transcripts
sudo chown -R $USER:$USER /data/webtoolbox
mkdir -p logs
```

4. Run with uvicorn:

```bash
source .venv/bin/activate
export WEBTOOLBOX_DATA_DIR=/data/webtoolbox
uvicorn webtoolbox.main:app --host 0.0.0.0 --port 8000 --app-dir src
```

## systemd service example

Create `/etc/systemd/system/webtoolbox.service`:

```ini
[Unit]
Description=Webtoolbox FastAPI Service
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/webtoolbox
Environment="WEBTOOLBOX_DATA_DIR=/data/webtoolbox"
Environment="WEBTOOLBOX_LOGS_DIR=/opt/webtoolbox/logs"
ExecStart=/opt/webtoolbox/.venv/bin/uvicorn webtoolbox.main:app --host 0.0.0.0 --port 8000 --app-dir src
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable webtoolbox
sudo systemctl start webtoolbox
sudo systemctl status webtoolbox
```

## Deployment via git pull

From the container:

```bash
cd /opt/webtoolbox
git pull
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart webtoolbox
```

## Notes on transcription engines

- Preferred: `faster-whisper` on CPU (`compute_type=int8`)
- Fallback: `openai-whisper` if `faster-whisper` import/model init fails

## Future tools

Add new tools under `src/webtoolbox/tools/<tool_name>/` and include each tool router in `main.py`.
