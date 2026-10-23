# GitHub Hub: Termux Universal AI

A self-bootstrapping AI app for Android Termux that:
- Shows a beautiful TUI-style menu
- Lets you choose a GGUF model
- Downloads the model directly from Hugging Face
- Runs in Termux either:
  - CLI chat mode
  - Local browser Web UI mode
- Tries to install required dependencies automatically
- Falls back safely if llama-cpp-python cannot build on the current Android device

This app is built to work inside Android Termux without manual pip setup.

---

## Features

- Terminal UI inspired by a clean AI launcher layout
- Model selection menu:
  - Qwen 2.5 0.5B
  - TinyLlama 1.1B
  - Qwen 2.5 1.5B
  - Llama 3.2 1B
  - Custom GGUF
- Direct GGUF download from Hugging Face
- Real-time progress bar while downloading
- CPU-only local inference support
- Browser-based chat UI via Flask
- Fully self-bootstrapping for basic setup
- Works on Android/Termux without pre-installing packages manually

---

## Requirements

### For Termux on Android
You need:
- Android device with Termux installed
- Working internet connection
- Basic Termux packages

### Recommended packages
Run this first:

```bash
pkg update && pkg upgrade -y
pkg install -y git python clang cmake make ninja
```

If `pkg install` fails for some package, you can try:

```bash
apt update && apt upgrade -y
apt install -y git python clang cmake make ninja
```

---

## Installation

1. Open Termux
2. Go to your project directory:

```bash
cd ~/ && mkdir -p Ai.py && cd Ai.py
```

3. Create `app.py` with the project code
4. Save it as:

```bash
app.py
```

Then run:

```bash
python app.py
```

or

```bash
python3 app.py
```

---

## How it works

When you run:

```bash
python app.py
```

the script does this automatically:

1. Checks if required packages exist
2. Installs missing packages such as:
   - requests
   - flask
3. Tries to install `llama-cpp-python`
4. Shows a menu with AI models
5. Downloads the GGUF model from Hugging Face
6. Lets you choose:
   - Terminal CLI chat
   - Local web UI
7. Runs the model locally on CPU if possible

---

## Model menu

The app shows a menu like this:

- [1] Qwen 2.5 0.5B
- [2] TinyLlama 1.1B
- [3] Qwen 2.5 1.5B
- [4] Llama 3.2 1B
- [5] Custom GGUF

Example:
```bash
Select model number [1-5]: 2
```

Then the app downloads the matching GGUF model from Hugging Face.

---

## Download behavior

The app downloads a file directly using HTTP requests, for example:

```text
https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

It saves the file in:

```bash
~/.termux_universal_ai/models/
```

Example:

```bash
/data/data/com.termux/files/home/.termux_universal_ai/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

---

## Execution modes

### 1. Terminal CLI Chat
This runs inside the Termux terminal.

Example:
```bash
Choose mode [1-3]: 1
```

Then:

```text
You> hello
AI> Hello! I am the Termux Universal AI...
```

---

### 2. Local Web UI
This starts a browser UI on your Android device.

Example:
```bash
Choose mode [1-3]: 2
```

Then the app starts a local web server:

```text
http://127.0.0.1:7860
```

Open the browser on the same device and use the UI.

---

## Important notes about llama-cpp-python

In some Termux builds, `llama-cpp-python` may fail to compile because Android toolchain support is incomplete. If that happens, the script automatically enters fallback mode instead of crashing.

That means:
- GGUF download still works
- app UI still runs
- but real model inference may not work until the backend is successfully installed

If backend install is successful, the app uses real local CPU inference.

---

## How to run the app

From Termux:

```bash
cd ~/Ai.py
python app.py
```

If Python points to the wrong version:

```bash
python3 app.py
```

If the app uses fallback mode, try installing backend tools manually:

```bash
pkg install -y clang cmake make ninja
python -m pip install --upgrade pip setuptools wheel
python -m pip install llama-cpp-python
```

If still failing, then the Android/Termux environment may not support a full native build for `llama-cpp-python` at the moment.

---

## Troubleshooting

### Error: huggingface_hub failed
This happens because some versions of `huggingface_hub` try to build `hf-xet` and need Rust/Android support that is not available in many Termux builds.

This project avoids that problem by downloading GGUF files directly with `requests` instead of using `huggingface_hub`.

---

### Error: No module named 'llama_cpp'
This means the backend is missing.

Try:

```bash
python -m pip install llama-cpp-python
```

If that fails, it likely means the current Termux environment cannot compile the native package.

---

### Error: Model download failed
Check:
- internet connection
- correct repo name
- correct GGUF filename
- selected model index

---

### Error: Flask not installed
The app tries to install it automatically, but if needed you can run:

```bash
python -m pip install flask
```

---

## Safe fallback mode

If `llama-cpp-python` is unavailable, the script still works in a safe fallback mode:
- model still downloads
- UI still opens
- chat still responds with a basic local answer
- no crash

This keeps the app usable while the backend is not available.

---

## Project files

### app.py
Main script that runs the app

### README.md
Project documentation and how-to instructions

---

## Example full workflow

```bash
pkg update && pkg upgrade -y
pkg install -y git python clang cmake make ninja

cd ~/ && mkdir -p Ai.py && cd Ai.py

python app.py
```

Then:
1. Choose model number
2. Wait for GGUF download
3. Choose:
   - 1 for CLI
   - 2 for Web UI
4. Chat with the model

---

## License

This project is open for personal and educational use.

You can modify and reuse it in Termux-based AI projects.

---

## Notes

This app is intended for:
- local experimentation
- learning AI on Android
- Termux-based prototype projects
- lightweight local GGUF model testing

It is not a cloud AI service and does not require external API keys.

---

## Quick summary

To run:

```bash
pkg install -y git python clang cmake make ninja
cd ~/Ai.py
python app.py
```

Then:
- select a model
- download model
- choose CLI or Web UI

---
