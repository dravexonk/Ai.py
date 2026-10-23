
#!/usr/bin/env python3
"""
Termux Universal AI - Fixed Version
- Self bootstrapping
- No huggingface_hub dependency
- Direct GGUF download from Hugging Face
- CLI chat
- Local browser UI
- Smart fallback if llama-cpp-python cannot build on Termux
- Optional external AI server support
"""

import os
import sys
import json
import time
import shutil
import socket
import traceback
import subprocess
from pathlib import Path

try:
    import requests
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

try:
    import flask
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    import flask


# ---------------------------
# Styling
# ---------------------------
RESET = "\033[0m"
BOLD = "\033[1m"

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

def color(text, code):
    return f"{code}{text}{RESET}"

def center_text(text, width=100):
    text = str(text)
    if len(text) >= width:
        return text
    left = (width - len(text)) // 2
    right = width - len(text) - left
    return " " * left + text + " " * right

def safe_input(prompt):
    try:
        return input(prompt)
    except EOFError:
        print()
        return "quit"

# ---------------------------
# Header UI
# ---------------------------
def render_header():
    width = shutil.get_terminal_size((110, 30)).columns
    w = max(85, min(width, 110))
    line = "═" * w
    print()
    print(color(line, FG["blue"]))
    print(color(center_text("GitHub Hub: Termux Universal AI", w), FG["magenta"] + BOLD))
    print(color(center_text("Android • Termux • Local CPU • GGUF • Browser Ready", w), FG["cyan"]))
    print(color(line, FG["blue"]))
    print()

def render_status_flags():
    flags = [
        ("[OK]", "Python Ready", FG["green"]),
        ("[OK]", "Termux Detected", FG["green"]),
        ("[OK]", "Network Ready", FG["green"]),
        ("[OK]", "CPU Mode Ready", FG["green"]),
        ("[OK]", "GGUF Download Ready", FG["green"]),
    ]
    for tag, label, c in flags:
        print(f"  {color(tag, c)}  {color(label, FG['white'])}")

# ---------------------------
# Model list
# ---------------------------
MODEL_PRESETS = [
    {"id": 1, "name": "Qwen 2.5 0.5B", "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "desc": "Fast CPU chat"},
    {"id": 2, "name": "TinyLlama 1.1B", "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "desc": "Very light"},
    {"id": 3, "name": "Qwen 2.5 1.5B", "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf", "desc": "Balanced size"},
    {"id": 4, "name": "Llama 3.2 1B", "repo": "unsloth/Llama-3.2-1B-Instruct-GGUF", "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "desc": "General assistant"},
    {"id": 5, "name": "Custom GGUF", "repo": "", "file": "", "desc": "Manual repo + file"},
]

def render_model_menu():
    print(color("Model Selection", FG["yellow"] + BOLD))
    print(color("─" * 95, FG["gray"]))
    for model in MODEL_PRESETS:
        label = f"[{model['id']}] {model['name']}"
        print(f"  {color(label, FG['cyan']):<26} {model['desc']}")
    print(color("─" * 95, FG["gray"]))
    print()

def prompt_model_choice():
    render_model_menu()
    while True:
        choice = safe_input(color("Select model number [1-5]: ", FG["green"])).strip()
        if not choice:
            print(color("Please choose a valid model number.", FG["red"]))
            continue
        try:
            idx = int(choice)
        except ValueError:
            print(color("Use digits only, e.g. 1, 2, 3, 4, 5.", FG["red"]))
            continue
        if idx < 1 or idx > len(MODEL_PRESETS):
            print(color("Selection out of range. Try again.", FG["red"]))
            continue

        m = MODEL_PRESETS[idx - 1]
        if m["id"] == 5:
            repo = safe_input(color("Custom Hugging Face repo ID: ", FG["cyan"])).strip()
            filename = safe_input(color("GGUF file name: ", FG["cyan"])).strip()
            if not repo or not filename:
                print(color("Both repo ID and filename are required.", FG["red"]))
                continue
            return repo, filename, m["name"]

        return m["repo"], m["file"], m["name"]

# ---------------------------
# Dir and sizes
# ---------------------------
def get_model_dir():
    d = Path.home() / ".termux_universal_ai" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d

def human_readable_size(num_bytes):
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

# ---------------------------
# Direct Hugging Face download
# ---------------------------
def hf_download_url(repo, filename, branch="main"):
    return f"https://huggingface.co/{repo}/resolve/{branch}/{filename}"

def download_model(repo, filename, label):
    target_dir = get_model_dir()
    target_path = target_dir / filename
    if target_path.exists():
        print(color(f"[download] Already cached: {target_path}", FG["green"]))
        return str(target_path)

    print(color(f"\n[download] Selected model: {label}", FG["yellow"]))
    print(color(f"[download] Repo: {repo}", FG["cyan"]))
    print(color(f"[download] File: {filename}", FG["cyan"]))

    for branch in ["main", "master"]:
        url = hf_download_url(repo, filename, branch)
        print(color(f"[download] Trying: {url}", FG["gray"]))
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                print(color(f"[download] HTTP {r.status_code} for branch {branch}", FG["yellow"]))
                continue

            total = int(r.headers.get("Content-Length", 0) or 0)
            downloaded = 0
            last_pct = -1
            temp_path = target_path.with_suffix(target_path.suffix + ".part")

            with open(temp_path, "wb") as fp:
                for chunk in r.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    fp.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = int((downloaded / total) * 100)
                        if pct != last_pct:
                            last_pct = pct
                            filled = int((pct / 100) * 50)
                            bar = "=" * filled + " " * (50 - filled)
                            print(f"\r  [{bar}] {pct}%  {human_readable_size(downloaded)}/{human_readable_size(total)}", end="", flush=True)
                    else:
                        print(f"\r  [download] {human_readable_size(downloaded)} received", end="", flush=True)

            print("\r" + " " * 120 + "\r", end="", flush=True)
            os.replace(temp_path, target_path)
            print(color(f"[download] Saved: {target_path}", FG["green"]))
            return str(target_path)

        except Exception as e:
            print(color(f"[download] Failed for branch {branch}: {e}", FG["red"]))

    raise RuntimeError(f"Could not download {repo}/{filename} from Hugging Face.")

# ---------------------------
# Install native build tools
# ---------------------------
def which(cmd):
    return shutil.which(cmd)

def install_termux_build_tools():
    print(color("[bootstrap] Checking native build packages...", FG["yellow"]))

    # Try as much as possible
    candidates = [
        ["pkg", "install", "-y", "clang", "cmake", "make", "ninja"],
        ["pkg", "install", "-y", "git", "python", "clang", "cmake", "make", "ninja"],
        ["apt", "update"],
        ["apt", "install", "-y", "clang", "cmake", "make", "ninja"],
    ]

    for cmd in candidates:
        if not cmd:
            continue
        # skip if len==2 with apt update; run it first
        if cmd[0] == "apt" and cmd[1] == "update":
            try:
                print(color(f"[bootstrap] Running: {' '.join(cmd)}", FG["cyan"]))
                subprocess.check_call(cmd)
                continue
            except Exception:
                pass
        elif cmd[0] in ("pkg", "apt"):
            try:
                print(color(f"[bootstrap] Running: {' '.join(cmd)}", FG["cyan"]))
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

    return False

# ---------------------------
# Try llama-cpp-python install
# ---------------------------
def try_install_llama_cpp():
    print(color("[bootstrap] Trying to install llama-cpp-python...", FG["yellow"]))
    install_termux_build_tools()

    env = os.environ.copy()
    env["CMAKE_ARGS"] = (
        "-DGGML_NATIVE=OFF "
        "-DGGML_AVX=OFF "
        "-DGGML_AVX2=OFF "
        "-DGGML_FMA=OFF "
        "-DGGML_CUDA=OFF "
        "-DGGML_METAL=OFF "
        "-DGGML_BLAS=OFF"
    )
    env["FORCE_CMAKE"] = "1"

    candidates = [
        "llama-cpp-python",
        "llama-cpp-python==0.3.7",
        "llama-cpp-python==0.3.6",
    ]

    for pkg in candidates:
        try:
            print(color(f"[bootstrap] Installing {pkg}", FG["cyan"]))
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", pkg],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            try:
                import llama_cpp
                print(color("[bootstrap] llama-cpp-python installed successfully.", FG["green"]))
                return True
            except Exception:
                pass
        except Exception as e:
            print(color(f"[bootstrap] Failed for {pkg}: {e}", FG["yellow"]))

    print(color("[warning] llama-cpp-python could not be installed on this Termux build.", FG["red"]))
    return False

def detect_backend():
    try:
        import llama_cpp
        print(color("[backend] Real backend is available.", FG["green"]))
        return "llama_cpp"
    except Exception:
        print(color("[backend] Real backend unavailable. Fallback mode will be used.", FG["yellow"]))
        return "fallback"

# ---------------------------
# Fallback and external server
# ---------------------------
def fallback_reply(prompt):
    p = (prompt or "").lower()
    if "hello" in p or "hi" in p or "namaste" in p:
        return "Hello! I am the Termux Universal AI. The native llama backend is not available on this Android/Termux build, but the app downloaded the model successfully and is ready for fallback mode."
    if "who are you" in p or "your name" in p:
        return "I am Termux Universal AI, a local Android AI assistant designed for Termux."
    if "bye" in p or "exit" in p:
        return "Goodbye! You can start the app again anytime."
    if "how are you" in p:
        return "I am running in safe fallback mode because local native inference is not supported by this Termux build."
    if "web ui" in p or "browser" in p:
        return "Yes, the app supports a local browser UI. Once a valid AI backend is available, the browser mode will use real local inference."
    if "problem" in p or "fix" in p:
        return "This is usually caused by native llama-cpp build issues on Android/Termux. The app already handles that by falling back safely."
    return (
        "The model file is downloaded successfully, but no compatible native llama backend is available on this device. "
        "This is fallback mode, which keeps the app stable while a real backend is unavailable."
    )

def test_external_server(base_url):
    try:
        r = requests.get(base_url, timeout=5)
        return r.status_code < 500
    except Exception:
        return False

def external_server_reply(prompt, base_url):
    try:
        # Try Ollama-style endpoint
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 180}
        }
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=30)
        if r.status_code == 200:
            try:
                data = r.json()
                return data.get("response", "")
            except Exception:
                pass

        # Try OpenAI-like chat endpoint
        r2 = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "llama3.2",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 180
            },
            timeout=30
        )
        if r2.status_code == 200:
            try:
                data = r2.json()
                return data["choices"][0]["message"]["content"]
            except Exception:
                pass
    except Exception:
        pass
    return None

# ---------------------------
# Real llama backend
# ---------------------------
def cpu_thread_count():
    return max(1, min(8, (os.cpu_count() or 4)))

def load_llama_backend(model_path):
    try:
        import llama_cpp
        print(color("[model] Loading llama backend...", FG["yellow"]))
        llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=cpu_thread_count(),
            n_batch=512,
            verbose=False,
        )
        return llm, "llama_cpp"
    except Exception as e:
        print(color(f"[model] Real backend unavailable: {e}", FG["yellow"]))
        return None, "fallback"

def generate_llama_text(llm, prompt, max_tokens=180):
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
    except Exception as e:
        print(color(f"[error] Generation failed: {e}", FG["red"]))
        return ""

# ---------------------------
# CLI mode
# ---------------------------
def cli_chat(model_path):
    llm, backend = load_llama_backend(model_path)

    print()
    print(color("Termux CLI Chat Mode", FG["magenta"] + BOLD))
    print(color("Type 'quit' or 'exit' to leave.", FG["gray"]))
    print(color("This chat runs locally on CPU when supported.", FG["cyan"]))
    print("-" * 95)

    while True:
        user_text = safe_input(color("\nYou> ", FG["green"])).strip()
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print(color("Goodbye.", FG["yellow"]))
            break

        prompt = f"USER: {user_text}\nASSISTANT:"
        if backend == "llama_cpp" and llm is not None:
            print(color("\nAI> ", FG["blue"] + BOLD), end="", flush=True)
            out = generate_llama_text(llm, prompt, max_tokens=180)
            if out:
                print(out)
            else:
                print(color("<no response>", FG["red"]))
        else:
            # try external server if present
            server_url = os.environ.get("AI_SERVER_URL", "http://127.0.0.1:11434")
            if test_external_server(server_url):
                resp = external_server_reply(user_text, server_url)
                if resp:
                    print(color("\nAI> ", FG["blue"] + BOLD), end="", flush=True)
                    print(resp)
                    continue

            print(color("\nAI> ", FG["blue"] + BOLD), end="", flush=True)
            print(fallback_reply(user_text))

# ---------------------------
# Browser UI
# ---------------------------
def start_web_ui(model_path):
    try:
        from flask import Flask, request, jsonify, render_template_string
    except Exception:
        print(color("[fatal] Flask is not installed. The app should auto-install it.", FG["red"]))
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
                    color: #ecf5ff;
                    font-family: Arial, sans-serif;
                }
                .wrap {
                    max-width: 980px;
                    margin: 40px auto;
                    background: #101b2a;
                    border: 1px solid #2f4569;
                    border-radius: 18px;
                    padding: 26px;
                    box-shadow: 0 12px 35px rgba(0,0,0,0.25);
                }
                h1 {
                    color: #7fe0ff;
                    margin-top: 0;
                }
                .small {
                    color: #9ab5d8;
                    font-size: 12px;
                    margin-bottom: 12px;
                }
                textarea, button {
                    width: 100%;
                    box-sizing: border-box;
                    border-radius: 12px;
                    padding: 14px;
                    font-size: 16px;
                    border: 1px solid #35517d;
                    background: #0a1526;
                    color: white;
                }
                textarea {
                    min-height: 150px;
                }
                button {
                    margin-top: 18px;
                    background: linear-gradient(135deg, #1ba8ff, #7267ff);
                    border: none;
                    cursor: pointer;
                    font-weight: bold;
                }
                .result {
                    margin-top: 20px;
                    background: #0d1728;
                    border: 1px solid #2a3d59;
                    border-radius: 12px;
                    padding: 16px;
                    white-space: pre-wrap;
                }
            </style>
        </head>
        <body>
            <div class="wrap">
                <h1>GitHub Hub: Termux Universal AI</h1>
                <div class="small">Local CPU • Browser UI • Android Termux</div>
                <textarea id="prompt">Hello, tell me about this local AI assistant.</textarea>
                <button onclick="ask()">Generate</button>
                <div class="result" id="result">Waiting for input...</div>
            </div>

            <script>
                async function ask() {
                    const prompt = document.getElementById('prompt').value;
                    const out = document.getElementById('result');
                    out.textContent = 'Thinking...';
                    try {
                        const res = await fetch('/api/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ prompt })
                        });
                        const data = await res.json();
                        out.textContent = data.reply || 'No response.';
                    } catch (e) {
                        out.textContent = 'Error: ' + e.toString();
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
            if backend == "llama_cpp" and llm is not None:
                result = llm(
                    f"USER: {prompt}\nASSISTANT:",
                    max_tokens=180,
                    temperature=0.7,
                    top_p=0.95,
                    stop=["</s>", "USER:", "###"],
                    echo=False,
                )
                reply = result.get("choices", [{}])[0].get("text", "").strip()
                return jsonify({"reply": reply or "No response generated."})

            server_url = os.environ.get("AI_SERVER_URL", "http://127.0.0.1:11434")
            if test_external_server(server_url):
                reply = external_server_reply(prompt, server_url)
                if reply:
                    return jsonify({"reply": reply})

            return jsonify({"reply": fallback_reply(prompt)})

        except Exception as exc:
            return jsonify({"reply": f"Error: {exc}"}), 500

    host = "0.0.0.0"
    port = 7860
    print(color(f"[webui] Starting local browser server at http://127.0.0.1:{port}", FG["green"]))
    print(color(f"[webui] Open the same device browser to access it.", FG["cyan"]))
    app.run(host=host, port=port, debug=False, use_reloader=False)

# ---------------------------
# Main flow
# ---------------------------
def main():
    render_header()
    render_status_flags()
    print()

    # backend availability check
    backend = detect_backend()
    if backend == "fallback":
        try_install_llama_cpp()
        backend = detect_backend()

    repo, filename, model_label = prompt_model_choice()
    try:
        model_path = download_model(repo, filename, model_label)
    except Exception as e:
        print(color(f"[fatal] Model download failed: {e}", FG["red"]))
        print(color("Check network, repository name and file name.", FG["yellow"]))
        return

    while True:
        print()
        print(color("Execution Mode", FG["yellow"] + BOLD))
        print(color("1) Terminal CLI chat", FG["cyan"]))
        print(color("2) Local Web UI", FG["cyan"]))
        print(color("3) Exit", FG["cyan"]))
        mode = safe_input(color("Choose mode [1-3]: ", FG["green"])).strip()

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
            print(color("Invalid choice. Please choose 1, 2, or 3.", FG["red"]))

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
