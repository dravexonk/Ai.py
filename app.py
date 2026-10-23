#!/usr/bin/env python3
# app.py
# Termux-friendly local web app:
# - auto-installs missing packages/dependencies
# - creates model storage folder
# - downloads GGUF models from direct URLs
# - serves local browser UI
# - runs local chat/inference once model is downloaded
#
# Run only:
#   python app.py

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, unquote

# -----------------------------
# 0) Logger
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------
# 1) Install Termux packages automatically
# -----------------------------
def run_cmd(cmd, timeout=600, allow_fail=False):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.stdout:
            logging.info(proc.stdout[:2000])
        if proc.stderr:
            logging.warning(proc.stderr[:2000])
        if proc.returncode != 0 and not allow_fail:
            raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
        return proc.returncode == 0
    except Exception as exc:
        if allow_fail:
            logging.warning("Command ignored because allow_fail=True: %s", exc)
            return False
        raise

def ensure_termux_packages():
    # Only if running inside Termux
    if not os.path.exists("/data/data/com.termux/files/home"):
        return True

    pkg = shutil.which("pkg")
    if not pkg:
        logging.warning("pkg not found. Skipping system package install.")
        return True

    required = ["git", "curl", "wget"]
    missing = [x for x in required if shutil.which(x) is None]

    if missing:
        logging.info("Installing required Termux packages: %s", ", ".join(missing))
        run_cmd([pkg, "update", "-y"], allow_fail=True)
        run_cmd([pkg, "install", "-y", *missing], allow_fail=True)

    # ensure Python exists
    if shutil.which("python") is None:
        run_cmd([pkg, "install", "-y", "python"], allow_fail=True)

    return True

# -----------------------------
# 2) Install Python pip dependencies automatically
# -----------------------------
def ensure_python_dependencies():
    packages = ["flask", "requests", "tqdm"]

    missing = []
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except Exception:
            missing.append(pkg)

    if not missing:
        return True

    logging.info("Installing missing Python dependencies: %s", ", ".join(missing))
    env = os.environ.copy()
    env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", *missing]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.stdout:
            logging.info(proc.stdout[:2000])
        if proc.stderr:
            logging.warning(proc.stderr[:2000])
        if proc.returncode != 0:
            logging.warning("Dependency installation returned non-zero exit code: %s", proc.returncode)
            return False
        return True
    except Exception as exc:
        logging.exception("Exception while installing Python dependencies: %s", exc)
        return False

# -----------------------------
# 3) Install dependencies before Flask imports
# -----------------------------
ensure_termux_packages()
ensure_python_dependencies()

try:
    import requests
    from flask import Flask, jsonify, redirect, render_template_string, request
except Exception as exc:
    logging.exception("Failed to import Flask/requests. Install manually: pip install flask requests tqdm")
    print("")
    print("Manual fix in Termux:")
    print("  pkg update -y")
    print("  pkg install -y git curl wget")
    print("  python -m pip install --upgrade pip setuptools wheel")
    print("  python -m pip install --upgrade flask requests tqdm")
    print("")
    sys.exit(1)

# -----------------------------
# 4) Model config and storage
# -----------------------------
APP_ROOT = Path(__file__).resolve().parent

MODEL_DIR = Path(os.environ.get("TERMUX_MODEL_DIR", str(Path.home() / "storage" / "shared" / "termux_models"))).resolve()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TASKS = {}
TASK_LOCK = threading.Lock()

ACTIVE_MODEL = {
    "name": None,
    "path": None,
    "source": None
}

# Recommended GGUF models (direct download URLs; safest for Termux)
RECOMMENDED_MODELS = [
    {
        "key": "tinyllama_1_1b",
        "name": "TinyLlama 1.1B Chat",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "description": "Very light; best for low-end Android"
    },
    {
        "key": "gemma_2b",
        "name": "Gemma 2 2B",
        "url": "https://huggingface.co/TheBloke/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it.Q4_K_M.gguf",
        "description": "Small but capable, good for general use"
    },
    {
        "key": "phi3_mini",
        "name": "Phi-3 Mini Instruct",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "description": "Efficient small model, very mobile-friendly"
    },
    {
        "key": "mistral_7b",
        "name": "Mistral 7B Instruct",
        "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "description": "Popular model, good if RAM is decent"
    },
    {
        "key": "llama3_1_8b",
        "name": "Llama 3.1 8B Instruct",
        "url": "https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "description": "Balanced general-purpose model"
    },
]

# -----------------------------
# 5) Helpers
# -----------------------------
def safe_filename(name):
    s = unquote((name or "model.gguf").strip())
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s[:200] or "model.gguf"
    return s

def set_task(task_id, status, message, progress=0.0, name=None, model_path=None):
    with TASK_LOCK:
        TASKS[task_id] = {
            "task_id": task_id,
            "status": status,
            "message": message,
            "progress": float(progress),
            "name": name,
            "model_path": model_path
        }

def list_downloaded_models():
    files = []
    for p in MODEL_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".gguf":
            files.append(str(p))
    return sorted(files)

def get_active_model_path():
    models = list_downloaded_models()
    if models:
        return models[0]
    return None

def parse_selection(form):
    selected_key = (form.get("model_key") or "").strip()
    custom_model = (form.get("custom_model") or "").strip()
    custom_name = (form.get("custom_filename") or "").strip()

    if selected_key:
        for model in RECOMMENDED_MODELS:
            if model["key"] == selected_key:
                return {"type": "url", "url": model["url"], "name": model["name"]}
        raise ValueError(f"Unknown model key: {selected_key}")

    if custom_model:
        custom_model = custom_model.strip()
        if custom_model.lower().startswith("http://") or custom_model.lower().startswith("https://"):
            return {
                "type": "url",
                "url": custom_model,
                "name": custom_name or safe_filename(urlparse(custom_model).path.split("/")[-1]) or "custom_model.gguf"
            }

        # They may provide only owner/repo: this app requires a direct GGUF URL for safest, zero-config operation
        raise ValueError(
            "Custom model input must be a direct GGUF download URL, for example:\n"
            "https://huggingface.co/owner/repo/resolve/main/model.gguf"
        )

    raise ValueError("No model selected.")

def download_file(url, target_path, on_progress=None):
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", "0") or "0")
    downloaded = 0

    with open(target_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            fh.write(chunk)
            downloaded += len(chunk)
            if on_progress:
                if total > 0:
                    pct = min(100.0, (downloaded / total) * 100.0)
                else:
                    pct = 0.0
                on_progress(pct)

def run_local_inference(prompt, model_path):
    # Try llama-cpp-python first
    try:
        import llama_cpp
        llm = llama_cpp.Llama(model_path=model_path, n_ctx=4096, verbose=False)
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
        )
        out = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if out:
            return out
    except Exception as exc:
        logging.warning("llama_cpp backend failed: %s", exc)

    # Fallback: local llama.cpp binaries
    candidates = ["llama-cli", "llama", "llama-server", "main", "./main"]
    for candidate in candidates:
        binary = shutil.which(candidate)
        if not binary:
            continue
        try:
            proc = subprocess.run(
                [binary, "--model", model_path, "--prompt", prompt, "--n-predict", "256", "--ctx-size", "4096"],
                capture_output=True,
                text=True,
                timeout=180
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if out.strip():
                return out.strip()
        except Exception as exc:
            logging.warning("CLI fallback failed for %s: %s", candidate, exc)

    raise RuntimeError(
        "No working local inference backend was found. "
        "Install llama-cpp-python or llama.cpp CLI manually for chat."
    )

# -----------------------------
# 6) Download worker
# -----------------------------
def download_worker(task_id, source):
    try:
        url = source["url"]
        model_name = source["name"]
        target = MODEL_DIR / safe_filename(model_name)

        if not target.exists():
            set_task(task_id, "downloading", f"Downloading {model_name}...", 0.0, model_name)

            def progress_cb(pct):
                with TASK_LOCK:
                    TASKS[task_id]["progress"] = float(pct)
                    TASKS[task_id]["message"] = f"Downloading {model_name}: {pct:.1f}%"

            download_file(url, str(target), progress_cb)
        else:
            logging.info("Model already exists: %s", target)

        if not target.exists():
            raise FileNotFoundError(f"File was not created: {target}")

        # mark it active
        global ACTIVE_MODEL
        ACTIVE_MODEL["name"] = model_name
        ACTIVE_MODEL["path"] = str(target)
        ACTIVE_MODEL["source"] = url

        with TASK_LOCK:
            TASKS[task_id]["status"] = "complete"
            TASKS[task_id]["progress"] = 100.0
            TASKS[task_id]["message"] = f"Model ready: {model_name}"
            TASKS[task_id]["model_path"] = str(target)

        logging.info("Download complete: %s", target)

    except Exception as exc:
        logging.exception("Download worker error: %s", exc)
        with TASK_LOCK:
            TASKS[task_id]["status"] = "error"
            TASKS[task_id]["message"] = str(exc)

# -----------------------------
# 7) HTML UI
# -----------------------------
INDEX_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Termux LLM Manager</title>
  <style>
    :root {
      --bg: #0b1120;
      --panel: rgba(17,24,39,0.95);
      --card: rgba(15,23,42,0.95);
      --text: #e5e7eb;
      --muted: #94a3b8;
      --line: rgba(255,255,255,0.12);
      --green: #34d399;
      --blue: #60a5fa;
      --red: #fca5a5;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: linear-gradient(180deg, #020817, var(--bg));
      color: var(--text);
      font-family: system-ui, -apple-system, sans-serif;
      padding: 24px;
    }
    .wrap { max-width: 1100px; margin: 0 auto; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
      margin-bottom: 24px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.25);
    }
    h1, h2, h3 { margin-top: 0; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
    }
    .card p {
      color: var(--muted);
      min-height: 56px;
      line-height: 1.45;
    }
    input, textarea, button {
      width: 100%;
      padding: 12px 14px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #111827;
      color: var(--text);
      font-size: 1rem;
      margin-top: 10px;
    }
    button {
      cursor: pointer;
      background: linear-gradient(90deg, var(--green), var(--blue));
      border: none;
      color: #06101b;
      font-weight: 800;
    }
    .progress {
      width: 100%;
      height: 16px;
      background: rgba(255,255,255,0.08);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 12px;
      border: 1px solid var(--line);
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--green), var(--blue));
      transition: width 0.2s ease;
    }
    .muted { color: var(--muted); font-size: 0.9rem; }
    .answer {
      margin-top: 12px;
      white-space: pre-wrap;
      min-height: 120px;
      background: rgba(15,23,42,0.95);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      line-height: 1.6;
    }
    .error { color: var(--red); font-weight: 700; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Termux LLM Manager</h1>
    <p class="muted">Download a GGUF model, then chat locally in the browser.</p>

    <div class="panel">
      <h2>1) Recommended models</h2>
      <div class="grid">
        {% for m in models %}
        <div class="card">
          <h3>{{ m.name }}</h3>
          <p>{{ m.description }}</p>
          <form method="post" action="/download">
            <input type="hidden" name="model_key" value="{{ m.key }}">
            <button type="submit">Download</button>
          </form>
        </div>
        {% endfor %}
      </div>
    </div>

    <div class="panel">
      <h2>2) Custom GGUF URL</h2>
      <form method="post" action="/download">
        <input type="text" name="custom_model" placeholder="Paste a direct GGUF URL (required)">
        <input type="text" name="custom_filename" placeholder="Optional custom filename">
        <button type="submit">Download custom model</button>
      </form>
    </div>

    <div class="panel">
      <h2>3) Download status</h2>
      {% if task_id %}
      <div id="status-message" class="muted">Loading...</div>
      <div class="progress"><div id="progress-bar" class="progress-bar" style="width:0%"></div></div>
      <div id="task-meta" class="muted">Task ID: {{ task_id }}</div>
      {% else %}
      <div class="muted">No active download task.</div>
      {% endif %}
      {% if active_model_name %}
      <div class="muted">Active model: {{ active_model_name }}</div>
      {% endif %}
    </div>

    <div class="panel">
      <h2>4) Chat in browser</h2>
      <textarea id="prompt" rows="7" placeholder="Ask something..."></textarea>
      <button id="chat-btn" type="button">Generate response</button>
      <div id="answer" class="answer">No response yet.</div>
    </div>
  </div>

  <script>
    const taskId = "{{ task_id or '' }}";

    function refreshStatus() {
      if (!taskId) return;
      fetch('/status?task_id=' + encodeURIComponent(taskId))
        .then(r => r.json())
        .then(data => {
          const msg = document.getElementById('status-message');
          const bar = document.getElementById('progress-bar');
          const meta = document.getElementById('task-meta');
          if (msg) msg.textContent = data.message || 'Working...';
          if (bar) bar.style.width = (data.progress || 0) + '%';
          if (meta) meta.textContent = 'Task ID: ' + (data.task_id || taskId) + ' • Status: ' + (data.status || 'unknown');
        })
        .catch(() => {});
    }

    if (taskId) {
      refreshStatus();
      setInterval(refreshStatus, 2000);
    }

    document.getElementById('chat-btn').addEventListener('click', function () {
      const prompt = document.getElementById('prompt').value.trim();
      const answerEl = document.getElementById('answer');
      if (!prompt) {
        answerEl.textContent = 'Please enter a prompt first.';
        return;
      }

      answerEl.textContent = 'Generating...';
      fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
      })
      .then(resp => resp.json())
      .then(data => {
        if (data.error) {
          answerEl.className = 'answer error';
          answerEl.textContent = 'Error: ' + data.error;
        } else {
          answerEl.className = 'answer';
          answerEl.textContent = data.response || 'No response generated.';
        }
      })
      .catch(err => {
        answerEl.className = 'answer error';
        answerEl.textContent = 'Chat request failed: ' + err;
      });
    });
  </script>
</body>
</html>
"""

# -----------------------------
# 8) Flask routes
# -----------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template_string(
        INDEX_TEMPLATE,
        models=RECOMMENDED_MODELS,
        task_id=None,
        active_model_name=ACTIVE_MODEL.get("name")
    )

@app.route("/download", methods=["POST"])
def download_route():
    try:
        source = parse_selection(request.form)
        task_id = uuid.uuid4().hex
        set_task(task_id, "queued", "Queued download", 0.0, source.get("name"))
        threading.Thread(target=download_worker, args=(task_id, source), daemon=True).start()
        return redirect(f"/progress?task_id={task_id}")
    except Exception as exc:
        logging.exception("Download route failed: %s", exc)
        return render_template_string(
            INDEX_TEMPLATE,
            models=RECOMMENDED_MODELS,
            task_id=None,
            active_model_name=ACTIVE_MODEL.get("name")
        ) + f"<div class='error' style='padding:12px;'>Error: {exc}</div>"

@app.route("/progress")
def progress_page():
    task_id = request.args.get("task_id", "").strip()
    if not task_id:
        return redirect("/")
    return render_template_string(
        INDEX_TEMPLATE,
        models=RECOMMENDED_MODELS,
        task_id=task_id,
        active_model_name=ACTIVE_MODEL.get("name")
    )

@app.route("/status")
def status_route():
    task_id = request.args.get("task_id", "").strip()
    if not task_id:
        return jsonify({"status": "idle", "message": "No active task", "progress": 0.0})

    task = TASKS.get(task_id, {})
    return jsonify({
        "task_id": task_id,
        "status": task.get("status", "idle"),
        "message": task.get("message", "No task found"),
        "progress": float(task.get("progress", 0.0)),
        "name": task.get("name"),
        "model_path": task.get("model_path")
    })

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        payload = request.get_json(silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Prompt cannot be empty."}), 400

        model_path = ACTIVE_MODEL.get("path") or get_active_model_path()
        if not model_path:
            return jsonify({"error": "No model downloaded yet. Select a model and wait for download to finish."}), 400

        answer = run_local_inference(prompt, model_path)
        return jsonify({"response": answer, "model_path": model_path})
    except Exception as exc:
        logging.exception("Chat request failed: %s", exc)
        return jsonify({"error": str(exc)}), 500

# -----------------------------
# 9) Startup
# -----------------------------
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))

    print("")
    print("============================================")
    print("Termux LLM Manager")
    print("============================================")
    print(f"Model storage: {MODEL_DIR}")
    print(f"Open in browser: http://localhost:{port}")
    print("Use only: python app.py")
    print("============================================")
    print("")

    app.run(host=host, port=port, debug=False, threaded=True)
