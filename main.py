#!/usr/bin/env python3
# app.py
# Zero-configuration Termux-friendly local model downloader + local chat UI
# Works on Android/Termux with Flask and llama-cpp-python if available.

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse, unquote

# --------------------------------------------------------------------
# 1) Dependency bootstrap before importing flask / huggingface_hub
# --------------------------------------------------------------------
def ensure_python_dependencies():
    required = ["flask", "requests", "huggingface_hub", "tqdm"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg.replace("-", "_"))
        except Exception:
            missing.append(pkg)

    if missing:
        logging.info("Installing missing Python dependencies: %s", ", ".join(missing))
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", *missing],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logging.exception("Failed to install required Python dependencies: %s", e)

    # Optional install: llama-cpp-python for local inference.
    try:
        __import__("llama_cpp")
        logging.info("llama-cpp-python is available.")
    except Exception:
        try:
            logging.info("Attempting to install optional package: llama-cpp-python")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "llama-cpp-python"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logging.warning("Could not install llama-cpp-python automatically: %s", e)
            logging.warning("The app will still work for downloads and will provide a clean fallback message during chat.")

ensure_python_dependencies()

# Now import app stack
import requests
from flask import Flask, jsonify, redirect, render_template_string, request, url_for
from huggingface_hub import snapshot_download

# --------------------------------------------------------------------
# 2) Global config and directories
# --------------------------------------------------------------------
LOGGING_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ROOT = Path(
    os.environ.get(
        "TERMUX_MODEL_DIR",
        str(Path.home() / "storage" / "shared" / "termux_models"),
    )
)
if not DEFAULT_MODEL_ROOT.exists():
    try:
        DEFAULT_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        DEFAULT_MODEL_ROOT = APP_ROOT / "models"
        DEFAULT_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = DEFAULT_MODEL_ROOT
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Track background tasks / model state
TASKS = {}
ACTIVE_MODEL = {
    "name": None,
    "path": None,
    "repo_id": None,
    "source_url": None,
}
TASK_LOCK = threading.Lock()
DOWNLOAD_EXECUTOR = ThreadPoolExecutor(max_workers=2)

# Recommended models (light/mobile friendly and popular reasonable options)
MODEL_CATALOG = [
    {
        "key": "llama3_1_8b",
        "name": "Llama 3.1 8B Instruct (mobile-friendly GGUF)",
        "repo_id": "unsloth/Meta-Llama-3.1-8B-Instruct-GGUF",
        "description": "Strong general-purpose model, good balance for Android/Termux",
    },
    {
        "key": "mistral_7b",
        "name": "Mistral 7B Instruct",
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "description": "Fast and capable, widely used 7B model",
    },
    {
        "key": "gemma_2_2b",
        "name": "Gemma 2 2B Instruct",
        "repo_id": "TheBloke/gemma-2-2b-it-GGUF",
        "description": "Very lightweight and good for lower-end Android devices",
    },
    {
        "key": "phi3_mini",
        "name": "Phi-3 Mini Instruct",
        "repo_id": "microsoft/Phi-3-mini-4k-instruct-gguf",
        "description": "Very good small model for mobile environments",
    },
    {
        "key": "tinyllama",
        "name": "TinyLlama 1.1B Chat",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "description": "Very small model, best for low-memory devices",
    },
]

# --------------------------------------------------------------------
# 3) Utility functions
# --------------------------------------------------------------------
def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "model.gguf")
    return cleaned[:200] or "model.gguf"

def determine_model_path_from_downloaded_repo(root_dir: Path):
    candidates = sorted(root_dir.rglob("*.gguf"))
    if not candidates:
        # Some repos may store .gguf in nested directories or use symlinks.
        candidates = sorted(root_dir.rglob("*.GGUF"))
    if not candidates:
        raise FileNotFoundError(f"No .gguf file found under {root_dir}")
    return str(candidates[0])

def choose_best_gguf(files):
    pick = None
    best_score = -999999
    for file in files:
        name = str(file).lower()
        score = 0
        if name.endswith(".gguf"):
            score += 100
        if "q4" in name:
            score += 25
        if "q5" in name:
            score += 20
        if "q8" in name:
            score += 15
        if "k_m" in name:
            score += 15
        if "instruct" in name:
            score += 10
        if "phi" in name:
            score += 8
        if "tinyllama" in name:
            score += 6
        if "f16" in name or "fp16" in name:
            score -= 10
        if "q2" in name:
            score -= 15
        if "q3" in name:
            score -= 5
        if "4bit" in name:
            score += 7
        if "8bit" in name:
            score += 5
        # Prefer smaller but not too tiny, but this is a rough heuristic.
        if score > best_score:
            best_score = score
            pick = file
    return str(pick or files[0])

def list_downloaded_models():
    results = []
    try:
        for file in sorted(MODEL_DIR.rglob("*.gguf")):
            if file.is_file():
                results.append(str(file))
        for file in sorted(MODEL_DIR.rglob("*.GGUF")):
            if file.is_file() and str(file) not in results:
                results.append(str(file))
    except Exception:
        pass
    return results

def resolve_repo_id(raw_value):
    value = (raw_value or "").strip()
    if not value:
        raise ValueError("No model was provided.")
    if value.lower().startswith("http://") or value.lower().startswith("https://"):
        return {"kind": "url", "url": value}
    if "/" not in value:
        raise ValueError("Model repo ID must look like 'owner/repo' or be a direct URL.")
    return {"kind": "hf", "repo_id": value}

def get_model_input_from_request(form):
    # Preferred UI flow: card buttons send model_key, custom input optional.
    custom = (form.get("custom_model") or "").strip()
    custom_filename = (form.get("custom_filename") or "").strip()
    selected_key = (form.get("model_key") or "").strip()

    if selected_key:
        for model in MODEL_CATALOG:
            if model["key"] == selected_key:
                return {
                    "kind": "hf",
                    "repo_id": model["repo_id"],
                    "filename": None,
                    "name": model["name"],
                }
        raise ValueError(f"Unknown recommended model key: {selected_key}")

    if custom:
        data = resolve_repo_id(custom)
        data["filename"] = custom_filename or None
        return data

    raise ValueError("Please select a model or enter a custom one.")

def find_gguf_file_for_hf_repo(repo_id: str, filename_hint: str = None):
    try:
        if filename_hint:
            # direct file name known
            from huggingface_hub import hf_hub_download
            final_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename_hint,
                local_dir=str(MODEL_DIR),
                local_dir_use_symlinks=False,
            )
            return final_path

        # Auto-select the best .gguf file from the repo
        local_snapshot = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(MODEL_DIR),
            allow_patterns=["*.gguf", "*.GGUF"],
            local_dir_use_symlinks=False,
            resume_download=True,
            token=None,
        )
        snapshot_path = Path(local_snapshot)
        files = []
        for f in snapshot_path.rglob("*"):
            if f.is_file() and f.suffix.lower() == ".gguf":
                files.append(f)
        if not files:
            raise FileNotFoundError(f"No GGUF model files were found in repository: {repo_id}")
        return str(choose_best_gguf(files))
    except Exception as exc:
        logging.exception("GGUF repo lookup failed for %s", repo_id)
        raise

def download_from_direct_url(url: str):
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid download URL: {url}")

    filename = safe_filename(unquote(parsed.path.split("/")[-1]) or "downloaded_model.gguf")
    target_path = MODEL_DIR / filename

    # If the file already exists, reuse it
    if target_path.exists():
        return str(target_path)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        progress = min(100.0, (downloaded / total) * 100.0)
                    else:
                        progress = 0.0
                    # status is updated in the caller around this function
                    yield progress

    return str(target_path)

def set_task(task_id, status, message, progress=0.0, model_path=None, name=None):
    with TASK_LOCK:
        TASKS[task_id] = {
            "task_id": task_id,
            "status": status,  # queued, downloading, complete, error
            "message": message,
            "progress": float(progress),
            "model_path": model_path,
            "name": name,
        }

def get_active_model_path():
    if ACTIVE_MODEL.get("path") and Path(ACTIVE_MODEL["path"]).exists():
        return ACTIVE_MODEL["path"]
    downloaded = list_downloaded_models()
    if downloaded:
        return downloaded[0]
    return None

def generate_local_response(prompt: str, model_path: str):
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError("The model file does not exist on disk.")

    # Try llama-cpp-python first.
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=0, verbose=False)
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.7,
        )
        answer = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if answer:
            return answer
        return "No response generated."
    except Exception as exc:
        logging.exception("llama_cpp generation path failed: %s", exc)

    # Fallback: if llama.cpp CLI is installed on the system, try it.
    binary_candidates = ["llama-cli", "llama", "llama-server", "main", "./main"]
    for candidate in binary_candidates:
        found = shutil.which(candidate)
        if found:
            try:
                cmd = [found, "--model", model_path, "--prompt", prompt, "--n-predict", "256", "--ctx-size", "4096"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                out = proc.stdout.strip() + "\n" + proc.stderr.strip()
                if out.strip():
                    return out.strip()
            except Exception as inner_exc:
                logging.warning("CLI fallback failed using %s: %s", candidate, inner_exc)
                continue

    raise RuntimeError(
        "No working local inference backend was found. "
        "The app tried llama-cpp-python and local llama.cpp binaries but none were available. "
        "You can still download and keep the model; to chat locally, install llama-cpp-python or llama.cpp."
    )

# --------------------------------------------------------------------
# 4) Background download workers
# --------------------------------------------------------------------
def download_model_task(task_id: str, source_data: dict):
    set_task(task_id, "downloading", "Preparing download...", 0.0)

    try:
        if source_data.get("kind") == "hf":
            repo_id = source_data.get("repo_id")
            filename = source_data.get("filename") or None
            model_name = source_data.get("name") or repo_id

            if not repo_id:
                raise ValueError("Repository ID is required.")

            set_task(task_id, "downloading", f"Downloading from Hugging Face: {repo_id}", 0.0, None, model_name)

            model_path = find_gguf_file_for_hf_repo(repo_id, filename)
            final_path = Path(model_path)

            # Some repos may download into nested directories; no issue.
            if final_path.exists():
                with TASK_LOCK:
                    TASKS[task_id]["status"] = "complete"
                    TASKS[task_id]["message"] = f"Model ready: {model_name}"
                    TASKS[task_id]["progress"] = 100.0
                    TASKS[task_id]["model_path"] = str(final_path)
                    TASKS[task_id]["name"] = model_name
                global ACTIVE_MODEL
                ACTIVE_MODEL["name"] = model_name
                ACTIVE_MODEL["path"] = str(final_path)
                ACTIVE_MODEL["repo_id"] = repo_id
                ACTIVE_MODEL["source_url"] = None
                logging.info("HF model downloaded to %s", final_path)
                return

            raise FileNotFoundError(f"Downloaded model file not found for repo: {repo_id}")

        elif source_data.get("kind") == "url":
            url = source_data.get("url")
            if not url:
                raise ValueError("No direct URL was provided.")
            model_name = source_data.get("name") or urlparse(url).path.split("/")[-1] or "downloaded_model.gguf"
            set_task(task_id, "downloading", f"Downloading model from direct URL: {url}", 0.0, None, model_name)

            total_bytes = 0
            try:
                with requests.get(url, stream=True, timeout=60) as r:
                    total_bytes = int(r.headers.get("Content-Length", "0") or "0")
                    filename = safe_filename(unquote((urlparse(url).path).split("/")[-1]) or "downloaded_model.gguf")
                    target_path = MODEL_DIR / filename
                    downloaded = 0
                    with open(target_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=65536):
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_bytes > 0:
                                progress = min(100.0, (downloaded / total_bytes) * 100.0)
                            else:
                                progress = 0.0
                            with TASK_LOCK:
                                TASKS[task_id]["progress"] = float(progress)
                                TASKS[task_id]["message"] = f"Downloading {filename}: {int(progress)}%"
                    if not target_path.exists():
                        raise FileNotFoundError("Downloaded file could not be written.")
            except requests.RequestException as exc:
                raise RuntimeError(f"Failed to fetch URL: {exc}") from exc

            with TASK_LOCK:
                TASKS[task_id]["status"] = "complete"
                TASKS[task_id]["message"] = f"Model ready: {model_name}"
                TASKS[task_id]["progress"] = 100.0
                TASKS[task_id]["model_path"] = str(MODEL_DIR / safe_filename(model_name))
                TASKS[task_id]["name"] = model_name

            target_path = MODEL_DIR / safe_filename(model_name)
            if not target_path.exists():
                target_path = MODEL_DIR / safe_filename(urlparse(url).path.split("/")[-1] or "downloaded_model.gguf")
            ACTIVE_MODEL["name"] = model_name
            ACTIVE_MODEL["path"] = str(target_path)
            ACTIVE_MODEL["repo_id"] = None
            ACTIVE_MODEL["source_url"] = url
            logging.info("Direct URL model downloaded to %s", target_path)
            return

        else:
            raise ValueError("Unsupported model source.")
    except Exception as exc:
        logging.exception("Model download task failed: %s", exc)
        with TASK_LOCK:
            TASKS[task_id]["status"] = "error"
            TASKS[task_id]["message"] = str(exc)
            TASKS[task_id]["progress"] = 0.0
            TASKS[task_id]["error"] = traceback.format_exc()

# --------------------------------------------------------------------
# 5) Flask app
# --------------------------------------------------------------------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Termux LLM Manager</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111827;
      --panel: #1f2937;
      --panel-2: #0f172a;
      --text: #e5e7eb;
      --muted: #a1a1aa;
      --accent: #60a5fa;
      --accent-2: #34d399;
      --danger: #f87171;
      --border: rgba(255,255,255,0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, sans-serif;
      background: linear-gradient(180deg, var(--bg), #020817);
      color: var(--text);
      min-height: 100vh;
      padding: 24px;
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
    }
    h1 { margin-bottom: 8px; }
    .subtitle { color: var(--muted); margin-bottom: 24px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
      margin-bottom: 28px;
    }
    .card, .panel {
      background: rgba(31,41,55,0.92);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 8px 18px rgba(0,0,0,0.2);
    }
    .card h3 {
      margin: 0 0 10px 0;
      font-size: 1.1rem;
    }
    .card p {
      color: var(--muted);
      min-height: 42px;
      margin: 0 0 14px 0;
      line-height: 1.4;
    }
    .card button, .primary, .secondary {
      border: none;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 0.95rem;
      cursor: pointer;
      transition: 0.15s ease;
    }
    .card button, .primary {
      background: var(--accent);
      color: #08111d;
      font-weight: 700;
    }
    .secondary {
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--border);
    }
    .panel {
      margin-bottom: 24px;
    }
    .custom-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    input[type="text"], textarea {
      width: 100%;
      padding: 12px 14px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      font-size: 1rem;
      resize: vertical;
    }
    .progress {
      width: 100%;
      height: 14px;
      background: rgba(255,255,255,0.08);
      border-radius: 999px;
      overflow: hidden;
      margin: 12px 0 8px 0;
      border: 1px solid var(--border);
    }
    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--accent-2), var(--accent));
      width: 0%;
      transition: width 0.35s ease;
    }
    .status-text {
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 12px;
    }
    .chat-box {
      width: 100%;
      min-height: 140px;
    }
    .answer-box {
      margin-top: 12px;
      padding: 16px;
      background: rgba(15,23,42,0.8);
      border: 1px solid var(--border);
      border-radius: 12px;
      min-height: 120px;
      white-space: pre-wrap;
      line-height: 1.5;
      color: #e2e8f0;
    }
    .small {
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 8px;
    }
    .error { color: var(--danger); }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Termux LLM Manager</h1>
    <div class="subtitle">
      Download a GGUF model from Hugging Face or a direct URL, then chat locally with llama-cpp-python (or a local llama.cpp binary if present).
    </div>

    <div class="panel">
      <h2>1) Download a model</h2>
      <div class="grid">
        {% for model in models %}
        <div class="card">
          <h3>{{ model.name }}</h3>
          <p>{{ model.description }}<br><span class="small">{{ model.repo_id }}</span></p>
          <form method="POST" action="/download">
            <input type="hidden" name="model_key" value="{{ model.key }}">
            <button type="submit">Download</button>
          </form>
        </div>
        {% endfor %}
      </div>

      <form method="POST" action="/download">
        <label for="custom_model"><strong>Custom model:</strong></label>
        <div class="custom-row">
          <input id="custom_model" type="text" name="custom_model" placeholder="owner/repo or direct https://... URL">
        </div>
        <div class="custom-row">
          <input type="text" name="custom_filename" placeholder="Optional GGUF file name (HF repos only)">
        </div>
        <div class="custom-row">
          <button class="primary" type="submit">Download custom model</button>
        </div>
      </form>
    </div>

    <div class="panel">
      <h2>2) Download status</h2>
      {% if task_id %}
      <div id="status-panel">
        <div class="status-text" id="status-message">Loading...</div>
        <div class="progress"><div id="progress-bar" class="progress-bar" style="width: 0%;"></div></div>
        <div id="status-meta" class="small">Task ID: {{ task_id }}</div>
      </div>
      {% else %}
      <div class="status-text">No active download task.</div>
      {% endif %}

      {% if active_model_name %}
      <div class="small" style="margin-top: 12px;">
        Active model: <strong>{{ active_model_name }}</strong>
      </div>
      {% endif %}
    </div>

    <div class="panel">
      <h2>3) Local chat</h2>
      <form id="chat-form">
        <textarea id="chat_prompt" class="chat-box" placeholder="Ask the model a question..."></textarea>
        <div class="custom-row">
          <button class="primary" type="submit">Generate response</button>
          <button class="secondary" type="button" id="clear-chat">Clear</button>
        </div>
      </form>
      <div id="answer-box" class="answer-box">No response yet. Download a model above and then ask a question.</div>
    </div>
  </div>

  <script>
    const taskId = "{{ task_id or '' }}";
    function updateStatus() {
      if (!taskId) return;
      fetch('/status?task_id=' + encodeURIComponent(taskId))
        .then(r => r.json())
        .then(data => {
          const msg = document.getElementById('status-message');
          const bar = document.getElementById('progress-bar');
          const meta = document.getElementById('status-meta');
          if (!msg || !bar || !meta) return;
          if (data && data.message) {
            msg.textContent = data.message;
          }
          const percent = Math.max(0, Math.min(100, Number(data.progress || 0)));
          bar.style.width = percent + '%';
          meta.textContent = 'Task ID: ' + (data.task_id || taskId) + ' • Status: ' + (data.status || 'unknown');
          if (data.status === 'complete') {
            msg.textContent = 'Complete: ' + (data.message || 'Model ready');
            clearInterval(statusTimer);
          } else if (data.status === 'error') {
            msg.innerHTML = '<span class=\"error\">Error: ' + (data.message || 'unknown error') + '</span>';
            clearInterval(statusTimer);
          }
        })
        .catch(() => {});
    }
    if (taskId) {
      updateStatus();
      const statusTimer = setInterval(updateStatus, 2000);
    }

    document.getElementById('chat-form').addEventListener('submit', function(e) {
      e.preventDefault();
      const prompt = document.getElementById('chat_prompt').value.trim();
      const answerBox = document.getElementById('answer-box');
      if (!prompt) {
        answerBox.textContent = 'Please enter a prompt first.';
        return;
      }
      answerBox.textContent = 'Generating...';
      fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
      })
      .then(r => r.json())
      .then(data => {
        if (data && data.error) {
          answerBox.textContent = 'Error: ' + data.error;
        } else {
          answerBox.textContent = data.response || 'No response generated.';
        }
      })
      .catch(err => {
        answerBox.textContent = 'Chat failed: ' + err;
      });
    });

    document.getElementById('clear-chat').addEventListener('click', function() {
      document.getElementById('chat_prompt').value = '';
      document.getElementById('answer-box').textContent = 'No response yet. Download a model above and then ask a question.';
    });
  </script>
</body>
</html>
"""

PROGRESS_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Downloading model...</title>
  <style>
    body {
      background: #0b1120;
      color: #e5e7eb;
      font-family: system-ui, -apple-system, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
    }
    .card {
      width: min(640px, 90vw);
      background: rgba(17,24,39,0.9);
      border: 1px solid rgba(255,255,255,0.14);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 32px rgba(0,0,0,0.35);
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 1.5rem;
    }
    .status { color: #cbd5e1; margin-bottom: 14px; }
    .progress {
      width: 100%;
      height: 18px;
      background: rgba(255,255,255,0.08);
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #34d399, #60a5fa);
      transition: width 0.2s ease;
    }
    .meta {
      margin-top: 10px;
      color: #94a3b8;
      font-size: 0.9rem;
    }
    a {
      display: inline-block;
      margin-top: 18px;
      color: #93c5fd;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Downloading model...</h1>
    <div id="status" class="status">Starting...</div>
    <div class="progress"><div id="progress-bar" class="progress-bar"></div></div>
    <div id="meta" class="meta">Task ID: {{ task_id }}</div>
    <a href="/">Back to model selection</a>
  </div>

  <script>
    const taskId = "{{ task_id }}";
    function poll() {
      fetch('/status?task_id=' + encodeURIComponent(taskId))
        .then(r => r.json())
        .then(data => {
          const status = document.getElementById('status');
          const bar = document.getElementById('progress-bar');
          const meta = document.getElementById('meta');
          if (!status || !bar || !meta) return;

          const message = data.message || 'Working...';
          const pct = Math.max(0, Math.min(100, Number(data.progress || 0)));
          status.textContent = message;
          bar.style.width = pct + '%';
          meta.textContent = 'Task ID: ' + (data.task_id || taskId) + ' • Status: ' + (data.status || 'unknown');

          if (data.status === 'complete') {
            setTimeout(() => { window.location.href = '/'; }, 1200);
          } else if (data.status === 'error') {
            setTimeout(() => { window.location.href = '/'; }, 2500);
          }
        })
        .catch(() => {});
    }
    poll();
    setInterval(poll, 2000);
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    active_name = ACTIVE_MODEL.get("name") or None
    return render_template_string(INDEX_HTML, models=MODEL_CATALOG, active_model_name=active_name, task_id=None)

@app.route("/download", methods=["POST"])
def download_route():
    try:
        source = get_model_input_from_request(request.form)
        task_id = uuid.uuid4().hex
        set_task(task_id, "queued", "Queued download...", 0.0)
        DOWNLOAD_EXECUTOR.submit(download_model_task, task_id, source)
        return redirect(f"/progress?task_id={task_id}")
    except Exception as exc:
        logging.exception("Download route failed: %s", exc)
        return render_template_string(
            INDEX_HTML,
            models=MODEL_CATALOG,
            active_model_name=ACTIVE_MODEL.get("name"),
            task_id=None,
        ) + f"<div style='color:red;padding:12px;'>Error: {exc}</div>"

@app.route("/progress")
def progress_page():
    task_id = request.args.get("task_id", "").strip()
    if not task_id:
        return redirect("/")
    return render_template_string(PROGRESS_HTML, task_id=task_id)

@app.route("/status")
def status_route():
    task_id = request.args.get("task_id", "").strip()
    if not task_id:
        return jsonify({"status": "idle", "message": "No active task.", "progress": 0.0})
    with TASK_LOCK:
        task = dict(TASKS.get(task_id, {}))
    if not task:
        return jsonify({"status": "idle", "message": "No task found.", "progress": 0.0, "task_id": task_id})
    return jsonify(task)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        payload = request.get_json(silent=True) or {}
        prompt = (payload.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "Prompt cannot be empty."}), 400

        model_path = ACTIVE_MODEL.get("path") or get_active_model_path()
        if not model_path:
            return jsonify({"error": "No model file is loaded yet. Download a model first."}), 400

        response = generate_local_response(prompt, model_path)
        return jsonify({"response": response, "model_path": model_path})
    except Exception as exc:
        logging.exception("Chat request failed: %s", exc)
        return jsonify({"error": str(exc)}), 500

# --------------------------------------------------------------------
# 6) Startup
# --------------------------------------------------------------------
if __name__ == "__main__":
    # Use port 5000 by default; change with PORT env var.
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    logging.info("Starting Termux LLM Manager on http://%s:%s", host, port)
    logging.info("Model storage directory: %s", MODEL_DIR)
    app.run(host=host, port=port, debug=False, threaded=True)
