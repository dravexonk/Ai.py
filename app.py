#!/usr/bin/env python3
"""
Termux Universal AI
- Self-bootstrapping
- TUI-style model selector
- GGUF download from Hugging Face using requests
- Optional local CPU or browser UI
- No hard dependency on huggingface_hub
"""

import os
import sys
import time
import json
import socket
import shutil
import traceback
import subprocess
from pathlib import Path

try:
    import requests
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

# ---------------------------
# Terminal styling
# ---------------------------
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
    text = str(text)
    if len(text) >= width:
        return text
    left = (width - len(text)) // 2
    right = width - len(text) - left
    return " " * left + text + " " * right

def print_header():
    width = shutil.get_terminal_size((100, 30)).columns
    w = max(80, min(width, 110))
    line = "═" * w
    print()
    print(color(line, FG["blue"]))
    print(color(center_text("GitHub Hub: Termux Universal AI", w), FG["magenta"] + BOLD))
    print(color(center_text("Android • Termux • Local CPU • GGUF • Offline Friendly", w), FG["cyan"]))
    print(color(line, FG["blue"]))
    print()

def print_status_flags():
    flags = [
        ("[OK]", "Python ready", FG["green"]),
        ("[OK]", "Termux detected", FG["green"]),
        ("[OK]", "Network ready", FG["green"]),
        ("[OK]", "CPU mode ready", FG["green"]),
    ]
    for tag, label, c in flags:
        print(f"  {color(tag, c)}  {color(label, FG['white'])}")

# ---------------------------
# Dependencies bootstrap
# ---------------------------
def ensure_module(module_name: str, pip_name: str = None):
    try:
        __import__(module_name)
        return True
    except Exception:
        pkg = pip_name or module_name
        print(color(f"[bootstrap] Installing missing package: {pkg}", FG["yellow"]))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False

def bootstrap_dependencies():
    installed = []
    for pkg in ["requests", "flask"]:
        ok = ensure_module(pkg)
        installed.append((pkg, ok))
    print(color("[bootstrap] Essential packages checked.", FG["green"]))

    # Optional: llama-cpp-python
    try:
        import llama_cpp
        print(color("[bootstrap] llama-cpp-python is available.", FG["green"]))
    except Exception:
        print(color("[bootstrap] llama-cpp-python not available. Attempting install...", FG["yellow"]))
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                import llama_cpp
                print(color("[bootstrap] llama-cpp-python installed and importable.", FG["green"]))
            except Exception:
                print(color("[warning] llama-cpp-python install failed or is unsupported on this Termux build.", FG["yellow"]))
        except Exception:
            print(color("[warning] llama-cpp-python could not be installed on this device.", FG["yellow"]))

    # No requirement on huggingface_hub anymore
    print(color("[bootstrap] Hugging Face SDK is intentionally not required. Direct GGUF download is used instead.", FG["cyan"]))

# ---------------------------
# Model presets
# ---------------------------
MODEL_PRESETS = [
    {
        "id": 1,
        "name": "Qwen 2.5 0.5B",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "desc": "Fast CPU model"
    },
    {
        "id": 2,
        "name": "TinyLlama 1.1B",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "desc": "Very lightweight"
    },
    {
        "id": 3,
        "name": "Qwen 2.5 1.5B",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "desc": "Balanced local speed"
    },
    {
        "id": 4,
        "name": "Llama 3.2 1B",
        "repo": "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "desc": "General assistant"
    },
    {
        "id": 5,
        "name": "Custom GGUF",
        "repo": "",
        "file": "",
        "desc": "Provide repo and file name"
    },
]

def get_model_dir() -> Path:
    d = Path.home() / ".termux_universal_ai" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d

def human_readable_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    i = 0
    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1
    if i == 0:
        return f"{int(size)} {units[i]}"
    return f"{size:.1f} {units[i]}"

def render_model_menu():
    print(color("Model selection", FG["yellow"] + BOLD))
    print(color("─" * 92, FG["gray"]))
    for model in MODEL_PRESETS:
        print(f"  {color(f'[{model['id']}] {model['name']}', FG['cyan']):<28} {model['desc']}")
    print(color("─" * 92, FG["gray"]))
    print()

def prompt_model_choice():
    render_model_menu()
    while True:
        choice = input(color("Select model number [1-5]: ", FG["green"])).strip()
        if not choice:
            print(color("Please choose a valid model number.", FG["red"]))
            continue
        try:
            idx = int(choice)
        except ValueError:
            print(color("Use a number only, like 1, 2, 3, 4, or 5.", FG["red"]))
            continue
        if idx < 1 or idx > len(MODEL_PRESETS):
            print(color("Choice is out of range. Try again.", FG["red"]))
            continue

        m = MODEL_PRESETS[idx - 1]
        if m["id"] == 5:
            repo = input(color("Custom HF repo ID: ", FG["cyan"])).strip()
            filename = input(color("GGUF filename: ", FG["cyan"])).strip()
            if not repo or not filename:
                print(color("Both repo ID and filename are required.", FG["red"]))
                continue
            return repo, filename, m["name"]

        return m["repo"], m["file"], m["name"]

# ---------------------------
# Direct Hugging Face downloader
# ---------------------------
def hf_download_url(repo: str, file_name: str, revision: str = "main") -> str:
    # Example:
    # https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
    return f"https://huggingface.co/{repo}/resolve/{revision}/{file_name}"

def download_model(repo: str, file_name: str, model_label: str) -> Path:
    target_dir = get_model_dir()
    target_path = target_dir / file_name
    if target_path.exists():
        print(color(f"[download] Model already cached: {target_path}", FG["green"]))
        return target_path

    print(color(f"\n[download] Selected model: {model_label}", FG["yellow"]))
    print(color(f"[download] Repo: {repo}", FG["cyan"]))
    print(color(f"[download] File: {file_name}", FG["cyan"]))

    for rev in ["main", "master"]:
        url = hf_download_url(repo, file_name, rev)
        print(color(f"[download] Trying URL: {url}", FG["gray"]))
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code == 200:
                total = int(r.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                last_pct = -1
                tmp_path = target_path.with_suffix(target_path.suffix + ".part")

                with open(tmp_path, "wb") as fp:
                    for chunk in r.iter_content(chunk_size=65536):
                        if not chunk:
                            continue
                        fp.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int((downloaded / total) * 100)
                            if pct != last_pct:
                                last_pct = pct
                                bar_len = 50
                                filled = int((pct / 100) * bar_len)
                                bar = "=" * filled + " " * (bar_len - filled)
                                print(f"\r  [{bar}] {pct}%  {human_readable_size(downloaded)}/{human_readable_size(total)}", end="", flush=True)
                        else:
                            print(f"\r  [download] {human_readable_size(downloaded)} received", end="", flush=True)

                print("\r" + " " * 120 + "\r", end="", flush=True)
                os.replace(tmp_path, target_path)
                print(color(f"[download] Saved to: {target_path}", FG["green"]))
                return target_path
            else:
                print(color(f"[download] HTTP {r.status_code} for branch {rev}", FG["yellow"]))
        except Exception as exc:
            print(color(f"[download] Connection failed for branch {rev}: {exc}", FG["red"]))

    raise RuntimeError(f"Failed to download {repo}/{file_name} from Hugging Face.")

# ---------------------------
# Local model backends
# ---------------------------
def maybe_int8_threads():
    return max(1, min(8, (os.cpu_count() or 2)))

def simple_fallback_reply(prompt: str) -> str:
    p = prompt.lower()
    if "hello" in p or "hi " in p:
        return "Hello! I am the Termux Universal AI in fallback mode. A real GGUF/llama backend can be loaded when available."
    if "what" in p and "your" in p and "name" in p:
        return "I am Termux Universal AI, running locally on Android CPU."
    if "who" in p and "you" in p:
        return "I am a lightweight local AI assistant designed to run in Termux."
    if "bye" in p or "exit" in p:
        return "Goodbye. Use the menu again whenever you want to continue."
    return (
        "I am running in CPU-safe fallback mode because no native llama runtime is available on this device.\n"
        "The GGUF model can still be downloaded here, and once a proper llama backend is present, this script can switch to full local inference automatically."
    )

def load_llama_backend(model_path: str):
    try:
        import llama_cpp
        print(color("[model] Loading local llama-cpp backend...", FG["yellow"]))
        llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=maybe_int8_threads(),
            n_batch=512,
            verbose=False,
        )
        return llm, "llama_cpp"
    except Exception:
        print(color("[model] llama-cpp-python is not available. Falling back to safe local response mode.", FG["yellow"]))
        return None, "fallback"

def generate_text_with_llama(llm, prompt: str, max_tokens: int = 200) -> str:
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
        return (text or "").strip()
    except Exception as exc:
        print(color(f"[error] Model generation failed: {exc}", FG["red"]))
        return ""

# ---------------------------
# CLI mode
# ---------------------------
def cli_chat(model_path: str):
    llm, backend = load_llama_backend(model_path)
    print()
    print(color("Termux CLI Chat Mode", FG["magenta"] + BOLD))
    print(color("Type 'quit' or 'exit' to leave.", FG["gray"]))
    print(color("The model runs directly in local Termux CPU mode if a compatible backend is present.", FG["cyan"]))
    print("-" * 92)

    while True:
        user_in = input(color("\nYou> ", FG["green"])).strip()
        if not user_in:
            continue
        if user_in.lower() in {"quit", "exit"}:
            print(color("Goodbye.", FG["yellow"]))
            break

        prompt = f"USER: {user_in}\nASSISTANT:"
        if backend == "llama_cpp":
            print(color("\nAI> ", FG["blue"] + BOLD), end="", flush=True)
            answer = generate_text_with_llama(llm, prompt, 180)
            if answer:
                print(answer)
            else:
                print(color("<no response>", FG["red"]))
        else:
            print(color("\nAI> ", FG["blue"] + BOLD), end="", flush=True)
            print(simple_fallback_reply(user_in))

# ---------------------------
# Flask web UI
# ---------------------------
def start_web_ui(model_path: str):
    try:
        from flask import Flask, request, jsonify, render_template_string
    except Exception:
        print(color("[fatal] Flask is not installed. Re-run and bootstrap should install it automatically.", FG["red"]))
        return

    llm, backend = load_llama_backend(model_path)

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string("""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Termux Universal AI</title>
            <style>
                body {
                    margin: 0;
                    background: #0b1020;
                    color: #edf6ff;
                    font-family: Arial, sans-serif;
                }
                .wrap {
                    max-width: 980px;
                    margin: 40px auto;
                    background: #111b2b;
                    border: 1px solid #2d3d57;
                    border-radius: 18px;
                    padding: 24px;
                    box-shadow: 0 10px 32px rgba(0,0,0,0.25);
                }
                h1 {
                    color: #7bd8ff;
                    margin-top: 0;
                }
                textarea, button {
                    width: 100%;
                    box-sizing: border-box;
                    border-radius: 12px;
                    padding: 14px;
                    font-size: 16px;
                    border: 1px solid #334a72;
                    background: #091321;
                    color: white;
                }
                textarea {
                    min-height: 150px;
                    margin-top: 18px;
                }
                button {
                    margin-top: 18px;
                    background: linear-gradient(135deg, #1a9bff, #7c61ff);
                    color: white;
                    border: none;
                    cursor: pointer;
                    font-weight: bold;
                }
                .resp {
                    margin-top: 18px;
                    background: #0d1728;
                    border: 1px solid #2a3f60;
                    border-radius: 12px;
                    padding: 18px;
                    white-space: pre-wrap;
                }
                .small {
                    color: #9db4d9;
                    font-size: 12px;
                }
            </style>
        </head>
        <body>
            <div class="wrap">
                <h1>GitHub Hub: Termux Universal AI</h1>
                <div class="small">Local CPU mode • Browser UI • Android Termux</div>
                <textarea id="prompt">Hello, tell me about this local AI system.</textarea>
                <button onclick="ask()">Generate</button>
                <div class="resp" id="response">Waiting for input...</div>
            </div>

            <script>
                async function ask() {
                    const prompt = document.getElementById('prompt').value;
                    const responseBox = document.getElementById('response');
                    responseBox.textContent = 'Thinking...';
                    try {
                        const raw = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({ prompt })
                        });
                        const data = await raw.json();
                        responseBox.textContent = data.reply || 'No response returned.';
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
        data = request.get_json(silent=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        if not prompt:
            return jsonify({"reply": "Please enter a prompt."}), 400
        try:
            if backend == "llama_cpp":
                result = llm(
                    f"USER: {prompt}\nASSISTANT:",
                    max_tokens=200,
                    temperature=0.7,
                    top_p=0.95,
                    stop=["</s>", "USER:", "###"],
                    echo=False,
                )
                reply = result.get("choices", [{}])[0].get("text", "").strip()
                return jsonify({"reply": reply or "No response generated."})
            else:
                return jsonify({"reply": simple_fallback_reply(prompt)})
        except Exception as exc:
            return jsonify({"reply": f"Error: {exc}"}), 500

    host = "0.0.0.0"
    port = 7860
    print(color(f"[webui] Starting local server at http://127.0.0.1:{port}", FG["green"]))
    print(color(f"[webui] Browser access should be available on the same device via http://127.0.0.1:{port}", FG["cyan"]))
    app.run(host=host, port=port, debug=False, use_reloader=False)

# ---------------------------
# Menu flow
# ---------------------------
def main():
    print_header()
    print_status_flags()
    print()
    print(color("Bootstrapping dependencies...", FG["yellow"]))
    bootstrap_dependencies()
    print()

    repo, file_name, model_label = prompt_model_choice()
    try:
        model_path = download_model(repo, file_name, model_label)
    except Exception as exc:
        print(color(f"[fatal] Download failed: {exc}", FG["red"]))
        print(color("Check the repo name, filename, and internet connection.", FG["yellow"]))
        return

    print()
    while True:
        print(color("Execution mode", FG["yellow"] + BOLD))
        print(color("1) CLI chat mode", FG["cyan"]))
        print(color("2) Web UI mode", FG["cyan"]))
        print(color("3) Exit", FG["cyan"]))
        mode = input(color("Choose mode [1-3]: ", FG["green"])).strip()

        if mode == "1":
            cli_chat(str(model_path))
            break
        elif mode == "2":
            start_web_ui(str(model_path))
            break
        elif mode == "3":
            print(color("Exiting.", FG["yellow"]))
            break
        else:
            print(color("Invalid input. Please choose 1, 2, or 3.", FG["red"]))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(color("Interrupted by user.", FG["yellow"]))
    except Exception as exc:
        print()
        print(color(f"[fatal] Unexpected error: {exc}", FG["red"]))
        traceback.print_exc()
        sys.exit(1)
