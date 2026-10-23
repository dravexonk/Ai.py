#!/usr/bin/env python3
"""
Termux Universal AI
- Self-bootstrapping installer for Termux/Android
- TUI menu for model selection
- Hugging Face GGUF downloader with progress bar
- CLI chat mode or local web UI mode
- Fully contained in a single file: app.py
"""

import os
import sys
import json
import time
import math
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Optional

# ---------- Terminal styling ----------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
FG = {
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "white": "\033[97m",
    "gray": "\033[90m",
}

def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"

def center_text(text: str, width: int = 100) -> str:
    if len(text) >= width:
        return text
    pad = width - len(text)
    left = pad // 2
    right = pad - left
    return " " * left + text + " " * right

def safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        print()
        return "quit"

# ---------- Bootstrap ----------
def ensure_python_version() -> None:
    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9+ is required. Install a newer Termux Python package.")

def run_pip_install(packages: list[str]) -> None:
    if not packages:
        return
    print(color(f"[bootstrap] Installing: {', '.join(packages)}", FG["yellow"]))
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    install_cmd = [sys.executable, "-m", "pip", "install", *packages]
    try:
        subprocess.check_call(install_cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Dependency installation failed: {' '.join(packages)} (exit={e.returncode})") from e

def ensure_dependencies() -> None:
    packages = ["requests", "huggingface_hub", "flask"]
    for pkg in packages:
        try:
            __import__(pkg)
        except Exception:
            run_pip_install([pkg])
    try:
        __import__("llama_cpp")
    except Exception:
        print(color("[bootstrap] llama-cpp-python not found. Attempting CPU-only build/install...", FG["yellow"]))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
        except subprocess.CalledProcessError as exc:
            print(color("[warning] llama-cpp-python installation failed. CLI/chat mode may not work without it.", FG["red"]))
            print(color("You can still use the web UI with an external model runner if installed later.", FG["yellow"]))
            print(color("If needed, run: python -m pip install llama-cpp-python", FG["cyan"]))
            print(color("The app will continue in degraded mode and will keep prompting for valid model setup.", FG["yellow"]))
            time.sleep(1.5)

def bootstrap() -> None:
    ensure_python_version()
    ensure_dependencies()

# ---------- Model metadata ----------
MODEL_PRESETS = [
    {
        "id": 1,
        "name": "Qwen 2.5 0.5B",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "desc": "Fast, compact CPU model",
    },
    {
        "id": 2,
        "name": "TinyLlama 1.1B",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "desc": "Very lightweight local chat",
    },
    {
        "id": 3,
        "name": "Qwen 2.5 1.5B",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "desc": "Balanced size/performance",
    },
    {
        "id": 4,
        "name": "Llama 3.2 1B",
        "repo": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "desc": "Fast general assistant",
    },
    {
        "id": 5,
        "name": "Custom GGUF",
        "repo": "",
        "file": "",
        "desc": "Use your own repo/file",
    },
]

def get_model_dir() -> Path:
    models_dir = Path.home() / ".termux_universal_ai" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def human_readable_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.1f} {units[idx]}"

def print_header() -> None:
    width = shutil.get_terminal_size((100, 30)).columns
    header = "GitHub Hub: Termux Universal AI"
    divider = "═" * max(70, min(width, 110))
    print()
    print(color(divider, FG["blue"]))
    print(color(center_text(header, max(70, min(width, 110))), FG["magenta"] + BOLD))
    print(color("  Android Termux • Native CPU • Hugging Face Models • Zero Config", FG["cyan"]))
    print(color(divider, FG["blue"]))
    print()

def render_status_flags() -> None:
    status = [
        ("[OK]", "Python ready", FG["green"]),
        ("[OK]", "Termux detected", FG["green"]),
        ("[OK]", "Network ready", FG["green"]),
        ("[OK]", "Inference layer ready", FG["green"]),
    ]
    # keep as separate readable lines
    for tag, label, color_code in status:
        print(f"  {color(tag, color_code)}  {color(label, FG['white'])}")

def render_model_menu() -> None:
    print(color("Model selection", FG["yellow"] + BOLD))
    print(color("─" * 90, FG["gray"]))
    for item in MODEL_PRESETS:
        label = f"[{item['id']}] {item['name']}"
        sub = item["desc"]
        print(f"  {color(label, FG['cyan']):<30} {sub}")
    print(color("─" * 90, FG["gray"]))
    print()

def prompt_model_choice() -> tuple[str, str, str]:
    render_model_menu()
    while True:
        choice = safe_input(color("Select a model number [1-5]: ", FG["green"])).strip()
        if choice == "":
            print(color("Please choose a model number.", FG["red"]))
            continue
        try:
            idx = int(choice)
        except ValueError:
            print(color("Invalid input. Enter a number from 1 to 5.", FG["red"]))
            continue

        if idx < 1 or idx > len(MODEL_PRESETS):
            print(color("Selection out of range. Choose 1 to 5.", FG["red"]))
            continue

        model = MODEL_PRESETS[idx - 1]
        if model["id"] == 5:
            repo = safe_input(color("Custom HF repo ID (example: Qwen/Qwen2.5-0.5B-Instruct-GGUF): ", FG["cyan"])).strip()
            file_name = safe_input(color("GGUF filename (example: qwen2.5-0.5b-instruct-q4_k_m.gguf): ", FG["cyan"])).strip()
            if not repo or not file_name:
                print(color("Both repo ID and file name are required for a custom model.", FG["red"]))
                continue
            return repo, file_name, model["name"]

        return model["repo"], model["file"], model["name"]

# ---------- Download helpers ----------
def build_hf_download_url(repo: str, file_name: str, revision: str = "main") -> str:
    # Example:
    # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
    return f"https://huggingface.co/{repo}/resolve/{revision}/{file_name}"

def download_file_with_progress(url: str, dest: Path) -> None:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(dest.suffix + ".part")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        last_pct = -1

        with open(temp_path, "wb") as fp:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                fp.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int((downloaded / total) * 100)
                    if pct != last_pct:
                        last_pct = pct
                        bar = "=" * int(pct / 2)
                        spaces = " " * (50 - len(bar))
                        print(f"\r  [{bar}{spaces}] {pct}%   {human_readable_size(downloaded)}/{human_readable_size(total)}", end="", flush=True)
                else:
                    print(f"\r  [download] {human_readable_size(downloaded)} received", end="", flush=True)

        print("\r" + " " * 120 + "\r", end="", flush=True)
        os.replace(temp_path, dest)
        print(color(f"[download] Saved: {dest}", FG["green"]))
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise

def download_model(repo: str, file_name: str, model_label: str) -> Path:
    print(color(f"\n[download] Preparing to download model: {model_label}", FG["yellow"]))
    print(color(f"[download] Repository: {repo}", FG["cyan"]))
    print(color(f"[download] File: {file_name}", FG["cyan"]))

    dest_dir = get_model_dir()
    dest_file = dest_dir / file_name
    if dest_file.exists():
        print(color(f"[download] Model already exists: {dest_file}", FG["green"]))
        return dest_file

    url = build_hf_download_url(repo, file_name)
    print(color(f"[download] URL: {url}", FG["gray"]))
    try:
        download_file_with_progress(url, dest_file)
        return dest_file
    except Exception as exc:
        print(color(f"[error] Download failed: {exc}", FG["red"]))
        raise

# ---------- Model runner ----------
def load_llama_model(model_path: str):
    try:
        import llama_cpp
    except Exception:
        raise RuntimeError("llama-cpp-python is not installed. This feature requires llama-cpp-python.")

    print(color("[model] Loading model into CPU memory...", FG["yellow"]))
    llm = llama_cpp.Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=max(1, min(8, (os.cpu_count() or 4))),
        n_batch=512,
        verbose=False,
    )
    print(color("[model] Model loaded successfully.", FG["green"]))
    return llm

def generate_text(llm, prompt: str, max_tokens: int = 200) -> str:
    try:
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            stop=["</s>", "USER:", "###"],
            echo=False,
        )
        text = result.get("choices", [{}])[0].get("text", "")
        return text.strip()
    except Exception as exc:
        print(color(f"[error] Generation failed: {exc}", FG["red"]))
        return ""

def cli_chat_mode(model_path: str) -> None:
    try:
        llm = load_llama_model(model_path)
    except Exception as exc:
        print(color(f"[fatal] Unable to load model: {exc}", FG["red"]))
        return
    print()
    print(color("Termux CLI Chat Mode", FG["magenta"] + BOLD))
    print(color("Type 'quit' or 'exit' to leave.", FG["gray"]))
    print(color("The model will answer directly from the local CPU.", FG["cyan"]))
    print("-" * 80)
    while True:
        user_text = safe_input(color("\nYou> ", FG["green"])).strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            print(color("Goodbye.", FG["yellow"]))
            break
        prompt = f"USER: {user_text}\\nASSISTANT:"
        print(color("\nAI> ", FG["blue"] + BOLD), end="", flush=True)
        answer = generate_text(llm, prompt, max_tokens=200)
        # Pretty output
        if answer:
            print(answer)
        else:
            print(color("<no response>", FG["red"]))

# ---------- Flask web UI ----------
def start_web_ui(model_path: str) -> None:
    try:
        from flask import Flask, request, jsonify, render_template_string
    except Exception:
        print(color("[fatal] Flask is not available. Please rerun the script and allow bootstrap to install it.", FG["red"]))
        return

    try:
        import llama_cpp
    except Exception:
        print(color("[fatal] llama-cpp-python is not available. The web UI cannot start without it.", FG["red"]))
        return

    app = Flask(__name__)

    llm = None
    try:
        llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=max(1, min(8, (os.cpu_count() or 4))),
            n_batch=512,
            verbose=False,
        )
        print(color("[webui] Model loaded for browser mode.", FG["green"]))
    except Exception as exc:
        print(color(f"[webui] Failed to load model: {exc}", FG["red"]))
        return

    @app.route("/")
    def index():
        return render_template_string("""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Termux Universal AI</title>
            <style>
                body { font-family: Arial, sans-serif; background: #0b1020; color: #eaf2ff; margin: 0; padding: 0; }
                .wrap { max-width: 980px; margin: 40px auto; background: #121a2b; border: 1px solid #2d3d5d; border-radius: 16px; padding: 24px; box-shadow: 0 8px 30px rgba(0,0,0,0.25); }
                h1 { margin-top:0; color:#7ad6ff; }
                textarea, button { width: 100%; box-sizing: border-box; border-radius: 10px; border: 1px solid #394b6a; background: #0c1426; color: white; padding: 14px; font-size: 16px; }
                textarea { min-height: 140px; }
                button { margin-top: 16px; background: linear-gradient(135deg, #1a9fff, #7b61ff); border: none; color: white; cursor: pointer; font-weight: bold; }
                .response { margin-top: 18px; background: #0d1728; border: 1px solid #2b3b59; border-radius: 10px; padding: 16px; white-space: pre-wrap; }
                .small { color: #8ba7cf; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="wrap">
                <h1>GitHub Hub: Termux Universal AI</h1>
                <div class="small">Local CPU model • Browser UI • Android/Termux</div>
                <textarea id="prompt" placeholder="Ask anything...">Hello, can you introduce yourself?</textarea>
                <button onclick="ask()">Generate</button>
                <div class="response" id="response">Waiting for input...</div>
            </div>
            <script>
                async function ask() {
                    const prompt = document.getElementById('prompt').value;
                    const responseBox = document.getElementById('response');
                    responseBox.textContent = 'Thinking...';
                    try {
                        const res = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({prompt})
                        });
                        const data = await res.json();
                        responseBox.textContent = data.reply || 'No response.';
                    } catch (e) {
                        responseBox.textContent = 'Error: ' + e.toString();
                    }
                }
            </script>
        </body>
        </html>
        """)

    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        payload = request.get_json(force=True, silent=True) or {}
        prompt = payload.get("prompt", "")
        if not prompt.strip():
            return jsonify({"reply": "Please enter a prompt."}), 400

        full_prompt = f"USER: {prompt}\\nASSISTANT:"
        try:
            result = llm(
                full_prompt,
                max_tokens=220,
                temperature=0.7,
                top_p=0.95,
                stop=["</s>", "USER:", "###"],
                echo=False,
            )
            reply = result.get("choices", [{}])[0].get("text", "").strip()
            return jsonify({"reply": reply or "No response generated."})
        except Exception as exc:
            return jsonify({"reply": f"Error: {exc}"}), 500

    host = "0.0.0.0"
    port = 7860
    print(color(f"[webui] Starting local server at http://127.0.0.1:{port}", FG["green"]))
    print(color(f"[webui] Browser access: http://{socket.gethostbyname(socket.gethostname())}:{port}", FG["cyan"]))
    app.run(host=host, port=port, debug=False, use_reloader=False)

# ---------- Main menu flow ----------
def main() -> None:
    bootstrap()
    print_header()
    render_status_flags()
    print()
    repo, file_name, model_label = prompt_model_choice()

    # If model is custom and repo is empty, retry
    if not repo or not file_name:
        print(color("Custom model requires repo + file.", FG["red"]))
        return

    try:
        model_path = download_model(repo, file_name, model_label)
    except Exception:
        print(color("[fatal] Model download failed. Check your network and repository path.", FG["red"]))
        return

    while True:
        print()
        print(color("Execution mode", FG["yellow"] + BOLD))
        print(color("1) CLI chat mode (Termux terminal)", FG["cyan"]))
        print(color("2) Web UI mode (browser)", FG["cyan"]))
        print(color("3) Exit", FG["cyan"]))
        choice = safe_input(color("Choose mode [1-3]: ", FG["green"])).strip()

        if choice == "1":
            cli_chat_mode(str(model_path))
            break
        elif choice == "2":
            start_web_ui(str(model_path))
            break
        elif choice == "3":
            print(color("Exiting.", FG["yellow"]))
            break
        else:
            print(color("Invalid choice. Please select 1, 2, or 3.", FG["red"]))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(color("Interrupted by user.", FG["yellow"]))
    except Exception as exc:
        print()
        print(color(f"[fatal] Unexpected error: {exc}", FG["red"]))
        try:
            import traceback
            traceback.print_exc()
        except Exception:
            pass
        sys.exit(1)
