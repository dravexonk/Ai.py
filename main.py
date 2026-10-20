#!/usr/bin/env python3

import os
import sys
import time
import shutil
import urllib.request
import subprocess
from pathlib import Path

# Qwen 2.5 0.5B GGUF model
REPO_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

HOME = Path.home()
BASE_DIR = HOME / "local-ai"
LLAMA_DIR = BASE_DIR / "llama.cpp"
BUILD_DIR = LLAMA_DIR / "build"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
SERVER_PATH = BUILD_DIR / "bin" / "llama-server"
LOG_PATH = BASE_DIR / "llama-server.log"

HOST = "127.0.0.1"
PORT = "8080"


def run(command, cwd=None):
    """Run a command and stop with a useful message if it fails."""
    print("+", " ".join(map(str, command)), flush=True)
    try:
        subprocess.run(command, cwd=cwd, check=True)
    except FileNotFoundError:
        print(f"\nCommand nahi mila: {command[0]}")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"\nCommand fail hua, exit code: {exc.returncode}")
        sys.exit(exc.returncode)


def is_termux():
    return (
        "com.termux" in os.environ.get("PREFIX", "")
        or Path("/data/data/com.termux/files/usr/bin/pkg").exists()
    )


def install_termux_packages():
    if not is_termux():
        print("Yeh auto-setup script Termux/Android ke liye hai.")
        print("Termux app ke andar `python main.py` chalayein.")
        sys.exit(1)

    pkg = shutil.which("pkg")
    if not pkg:
        print("Termux ka `pkg` command nahi mila.")
        sys.exit(1)

    print("\n[1/4] Termux packages check/install ho rahe hain...")
    run([pkg, "update", "-y"])
    run([pkg, "install", "-y", "git", "cmake", "clang", "make", "python"])


def build_llama_cpp():
    print("\n[2/4] llama.cpp check/build ho raha hai...")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if not (LLAMA_DIR / ".git").exists():
        run([
            "git", "clone",
            "--depth", "1",
            "https://github.com/ggml-org/llama.cpp",
            str(LLAMA_DIR),
        ])

    if SERVER_PATH.is_file():
        print("llama-server pehle se build hai; build skip kiya.")
        return

    run([
        "cmake",
        "-S", str(LLAMA_DIR),
        "-B", str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
    ])

    # -j2 RAM use ko kam rakhta hai; build phir bhi waqt le sakta hai.
    run([
        "cmake",
        "--build", str(BUILD_DIR),
        "--config", "Release",
        "--target", "llama-server",
        "-j2",
    ])

    if not SERVER_PATH.is_file():
        print(f"\nBuild hua, lekin server binary nahi mili:\n{SERVER_PATH}")
        sys.exit(1)


def download_model():
    print("\n[3/4] GGUF model check/download ho raha hai...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.is_file() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"Model pehle se maujood hai: {MODEL_PATH}")
        return

    temp_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
    request = urllib.request.Request(
        REPO_URL,
        headers={"User-Agent": "Termux-Local-AI/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0

            with open(temp_path, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        percent = downloaded * 100 // total
                        print(
                            f"\rDownload: {percent}% "
                            f"({downloaded // (1024 * 1024)} MB)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\rDownloaded {downloaded // (1024 * 1024)} MB",
                            end="",
                            flush=True,
                        )

        print()
        if temp_path.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded file bahut chhoti hai; model download verify karein.")

        temp_path.replace(MODEL_PATH)
        print(f"Model save hua: {MODEL_PATH}")

    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        print(f"\nModel download fail hua: {exc}")
        print("Internet connection aur Hugging Face model URL check karein.")
        sys.exit(1)


def start_server():
    print("\n[4/4] Local AI server start ho raha hai...")
    print(f"Browser mein kholein: http://{HOST}:{PORT}")
    print("Server band karne ke liye Ctrl+C dabayein.\n")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        str(SERVER_PATH),
        "-m", str(MODEL_PATH),
        "--host", HOST,
        "--port", PORT,
        "-c", "2048",
        "-ngl", "0",
    ]

    try:
        with open(LOG_PATH, "w", encoding="utf-8") as log_file:
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
    except KeyboardInterrupt:
        print("\nServer band kar diya.")
    except Exception as exc:
        print(f"\nServer start nahi hua: {exc}")
        print(f"Log file: {LOG_PATH}")
        sys.exit(1)


def main():
    install_termux_packages()
    build_llama_cpp()
    download_model()
    start_server()


if __name__ == "__main__":
    main()
