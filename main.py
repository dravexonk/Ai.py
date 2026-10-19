#!/usr/bin/env python3
"""
Termux Universal Local AI Engine (Advanced Multi-Model Interactive System)
--------------------------------------------------------------------------
Author: Modern AI Core
Features:
 - Pure CPU Execution for Android / Termux (No PyTorch needed)
 - Self-Healing C++ Compilation Engine for Termux ARM64 Architecture
 - Interactive Hugging Face Downloader & Custom Repo Direct Fetching
 - Multi-Turn Conversation Memory Buffer
 - Real-Time Token Generation & Streaming Output
"""

import os
import sys
import subprocess
import importlib.util

# ==========================================
# 1. ADVANCED AUTO-ENVIRONMENT SETUP & FIXER
# ==========================================

def run_cmd(command_list, suppress_output=False):
    """Executes shell command safely."""
    try:
        stdout_target = subprocess.DEVNULL if suppress_output else None
        stderr_target = subprocess.DEVNULL if suppress_output else None
        return subprocess.run(command_list, check=True, stdout=stdout_target, stderr=stderr_target)
    except Exception as e:
        return None

def verify_and_fix_ctransformers():
    """
    Termux missing C++ shared binary error fix.
    Force reinstalls ctransformers from source using Termux system compilers.
    """
    print("\n[!] CTransformers native binaries missing for Android platform.")
    print("[!] Initiating automatic local source compilation (Termux C++ Auto-Fix)...")
    print("[!] Installing native compilation toolchain (clang, cmake, ninja)...")
    
    # 1. Install system compilers via Termux pkg
    if os.path.exists("/data/data/com.termux/files/usr/bin/pkg"):
        run_cmd(["pkg", "install", "clang", "cmake", "make", "ninja", "python", "python-dev", "-y"])
    
    # 2. Reinstall ctransformers without binary isolation
    print("[!] Compiling ctransformers C++ engine locally for ARM64...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "ctransformers", "-y"])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ctransformers", "--no-build-isolation", "--no-binary", "ctransformers"
        ])
        print("\n[✔] CTransformers C++ core compiled successfully!\n")
    except Exception as err:
        print(f"\n[❌] Source compilation failed: {err}")
        print("[!] Attempting secondary fallback build flag...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "ctransformers", "--no-deps", "--force-reinstall"
            ])
        except Exception:
            print("\n[❌] Critical: Manual fix needed. Please run:")
            print("    pkg install clang cmake make -y")
            print("    pip install ctransformers --no-build-isolation --no-binary ctransformers")
            sys.exit(1)

def setup_environment():
    """
    Checks and automatically installs system packages via pkg (APT)
    and specific wheel-compatible PIP libraries for Termux.
    """
    print("\n[+] Checking System Environment for Termux & Linux...")
    
    # System package check (Termux pkg)
    if os.path.exists("/data/data/com.termux/files/usr/bin/pkg"):
        print("[+] Termux Environment Detected. Ensuring compiler binaries...")
        system_packages = ["clang", "cmake", "make", "ninja", "python", "python-pip", "git", "binutils"]
        for pkg in system_packages:
            try:
                subprocess.run(["pkg", "install", pkg, "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    # Python pip packages required
    required_pip_pkgs = [
        "ctransformers",
        "rich",
        "requests",
        "tqdm",
        "huggingface_hub<0.25.0"  # Prevents hf-xet/maturin rust build crashes
    ]

    missing = False
    for mod in ["ctransformers", "rich", "requests", "tqdm", "huggingface_hub"]:
        if importlib.util.find_spec(mod) is None:
            missing = True
            break

    if missing:
        print("[!] Missing Python libraries detected.")
        print("[!] Installing missing dependencies with Termux flags...\n")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation"] + required_pip_pkgs
            subprocess.check_call(cmd)
            print("\n[✔] Libraries installed successfully!\n")
        except Exception:
            try:
                cmd_fallback = [sys.executable, "-m", "pip", "install", "--prefer-binary"] + required_pip_pkgs
                subprocess.check_call(cmd_fallback)
            except Exception as e:
                print(f"\n[⚠️] Dependency installation warning: {e}")

# Execute Initial Setup
setup_environment()

# Import Installed Libraries Safely
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table
from huggingface_hub import hf_hub_download

console = Console()

# ==========================================
# 2. CONFIGURATION & PREDEFINED MODELS
# ==========================================

MODELS_DIRECTORY = "downloaded_models"
os.makedirs(MODELS_DIRECTORY, exist_ok=True)

PREDEFINED_MODELS = {
    "1": {
        "title": "Qwen 2.5 0.5B Instruct (Ultra Fast / Recommended for Termux)",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "model_type": "qwen2",
        "size": "~350 MB"
    },
    "2": {
        "title": "TinyLlama 1.1B Chat (Lightweight & Smart)",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "model_type": "llama",
        "size": "~630 MB"
    },
    "3": {
        "title": "Qwen 2.5 1.5B Instruct (High Accuracy / Multi-Lingual)",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "model_type": "qwen2",
        "size": "~1.1 GB"
    },
    "4": {
        "title": "Llama 3.2 1B Instruct (Meta's Small LLM)",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "model_type": "llama",
        "size": "~800 MB"
    }
}

SYSTEM_INSTRUCTION = (
    "You are an intelligent, helpful, and polite AI assistant. "
    "Respond accurately and clearly according to user preference in English or Hinglish."
)

# ==========================================
# 3. INTERACTIVE UI & DOWNLOAD MANAGEMENT
# ==========================================

def display_header():
    """Displays terminal ASCII Header Banner."""
    console.clear()
    banner = (
        "[bold magenta]🤖 TERMUX LOCAL AI ENGINE (OFFLINE CORE) 🤖[/bold magenta]\n\n"
        "[green]✔ Zero PyTorch / CPU-Only Native C++ Execution[/green]\n"
        "[green]✔ Interactive HuggingFace Downloader Built-In[/green]\n"
        "[green]✔ Multi-Turn Chat Memory & Streaming Tokens[/green]"
    )
    console.print(
        Panel(
            banner,
            title="[bold white]GitHub Hub: Termux Universal AI[/bold white]",
            border_style="bright_blue",
            expand=False
        )
    )

def download_hf_model(repo_id, filename):
    """Downloads model file from Hugging Face directly into local folder."""
    local_path = os.path.join(MODELS_DIRECTORY, filename)
    if os.path.exists(local_path):
        console.print(f"\n[bold green]✔ Model file exists locally: {filename}[/bold green]")
        return local_path

    console.print(f"\n[bold yellow]📥 Downloading from Hugging Face...[/bold yellow]")
    console.print(f"[cyan]Repo ID:[/cyan] {repo_id}")
    console.print(f"[cyan]File Name:[/cyan] {filename}\n")

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIRECTORY,
            local_dir_use_symlinks=False
        )
        console.print(f"\n[bold green]✔ Download Completed Successfully![/bold green]\n")
        return downloaded_file
    except Exception as error:
        console.print(f"\n[bold red]❌ Download Failed:[/bold red] {error}")
        if os.path.exists(local_path):
            os.remove(local_path)
        sys.exit(1)

def model_selection_menu():
    """Interactive menu for model choices."""
    display_header()
    
    table = Table(title="[bold cyan]Select AI Model to Download or Run[/bold cyan]", border_style="cyan")
    table.add_column("Option", style="bold yellow", justify="center")
    table.add_column("Model Name / Details", style="white")
    table.add_column("Approx Size", style="green", justify="center")

    for key, data in PREDEFINED_MODELS.items():
        table.add_row(key, data["title"], data["size"])
    
    table.add_row("C", "Custom HuggingFace GGUF Repo (Enter Custom Repo & File)", "Custom")
    
    console.print(table)

    choice = Prompt.ask(
        "\n[bold green]Select Option Number or 'C' for Custom[/bold green]",
        choices=list(PREDEFINED_MODELS.keys()) + ["C", "c"],
        default="1"
    )

    if choice.upper() == "C":
        console.print("\n[bold cyan]--- Custom HuggingFace GGUF Setup ---[/bold cyan]")
        repo_id = Prompt.ask("Enter HuggingFace Repo ID (e.g. Qwen/Qwen2.5-0.5B-Instruct-GGUF)")
        filename = Prompt.ask("Enter exact GGUF File Name (e.g. qwen2.5-0.5b-instruct-q4_k_m.gguf)")
        model_type = Prompt.ask("Enter Model Architecture Type (llama / qwen2 / mistral)", default="llama")
        
        file_path = download_hf_model(repo_id.strip(), filename.strip())
        return file_path, model_type.strip(), filename.strip()
    else:
        selected = PREDEFINED_MODELS[choice]
        file_path = download_hf_model(selected["repo_id"], selected["filename"])
        return file_path, selected["model_type"], selected["filename"]

# ==========================================
# 4. INFERENCE ENGINE & LOAD MANAGEMENT
# ==========================================

def initialize_model(model_path, model_type):
    """
    Loads quantized GGUF model with dynamic error catching and auto-repair.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]⏳ Loading GGUF Model into RAM...[/bold cyan]\n"
            f"[yellow]File Path: {model_path}[/yellow]\n"
            f"[green]Thread Optimization: Running on CPU Native Backend[/green]",
            title="[bold green]Model Loader[/bold green]",
            border_style="cyan"
        )
    )

    # Retry Mechanism to Auto-Fix Missing C++ Binaries
    for attempt in range(2):
        try:
            from ctransformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                gpu_layers=0,
                context_length=2048
            )
            return model
        except Exception as error:
            error_msg = str(error)
            if "Precompiled binaries are not available" in error_msg or "DLL load failed" in error_msg or attempt == 0:
                console.print(f"\n[bold yellow]⚠️ CTransformers Binary Error Detected:[/bold yellow] {error}")
                verify_and_fix_ctransformers()
            else:
                console.print(f"\n[bold red]❌ Failed to initialize CTransformers Model:[/bold red] {error}")
                sys.exit(1)

def run_chat_loop(model, model_filename):
    """Executes chat session with context buffer and streaming tokens."""
    console.clear()
    console.print(
        Panel(
            f"[bold magenta]🤖 AI SESSION ACTIVE ({model_filename})[/bold magenta]\n\n"
            "[green]Status: Offline Engine Ready[/green]\n"
            "[yellow]Commands:[/yellow]\n"
            "  • [cyan]/clear[/cyan] - Reset Conversation Memory\n"
            "  • [cyan]/help[/cyan]  - Show Available Commands\n"
            "  • [cyan]/exit[/cyan]  - Terminate Program",
            title="[bold white]Interactive Chat[/bold white]",
            border_style="bright_blue",
            expand=False
        )
    )

    chat_history = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold bright_blue]You[/bold bright_blue]")
            cleaned_input = user_input.strip()

            if not cleaned_input:
                continue

            cmd = cleaned_input.lower()

            if cmd in ["exit", "quit", "/exit", "bye"]:
                console.print("\n[bold yellow]AI: Goodbye! Have a great time.[/bold yellow]\n")
                break

            elif cmd in ["/clear", "clear"]:
                chat_history.clear()
                console.print("[bold green]✨ Conversation history cleared successfully![/bold green]")
                continue

            elif cmd in ["/help", "help"]:
                console.print(
                    Panel(
                        "[bold cyan]Control Commands:[/bold cyan]\n"
                        " 1. [yellow]/clear[/yellow] - Erase context memory.\n"
                        " 2. [yellow]/help[/yellow]  - Show menu details.\n"
                        " 3. [yellow]/exit[/yellow]  - Exit the program.",
                        title="Help Options",
                        border_style="cyan"
                    )
                )
                continue

            # Store user input
            chat_history.append({"role": "user", "content": cleaned_input})

            # Format ChatML Prompt Structure
            formatted_prompt = f"<|im_start|>system\n{SYSTEM_INSTRUCTION}<|im_end|>\n"
            for msg in chat_history:
                role = msg["role"]
                content = msg["content"]
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            formatted_prompt += "<|im_start|>assistant\n"

            console.print("\n[bold green]AI Response:[/bold green]", end=" ")
            full_response = ""

            # Real-Time Token Generation via Streaming
            try:
                stream = model(
                    formatted_prompt,
                    stream=True,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                for token in stream:
                    full_response += token
                    print(token, end="", flush=True)
                print()

            except Exception:
                raw_response = model(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                full_response = raw_response.strip()
                console.print(Markdown(full_response))

            # Save AI response into history
            chat_history.append({"role": "assistant", "content": full_response.strip()})

        except KeyboardInterrupt:
            console.print("\n\n[bold red]Session Interrupted by User.[/bold red]")
            break
        except Exception as error:
            console.print(f"\n[bold red]⚠️ Generation Error:[/bold red] {error}")

# ==========================================
# 5. MAIN EXECUTION ENTRY POINT
# ==========================================

def main():
    model_path, model_type, model_filename = model_selection_menu()
    loaded_model = initialize_model(model_path, model_type)
    run_chat_loop(loaded_model, model_filename)

if __name__ == "__main__":
    main()
