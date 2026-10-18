#!/usr/bin/env python3
"""
Termux Universal Local AI Engine (Advanced Multi-Model Interactive System)
--------------------------------------------------------------------------
Author: Modern AI Core
Features:
 - Pure CPU Execution for Android / Termux (No PyTorch needed)
 - Interactive Hugging Face Downloader & Custom Repo Direct Fetching
 - Automatic C++ System Binaries & Python Library Setup
 - Multi-Turn Conversation Memory Buffer
 - Real-Time Token Generation & Streaming Output
"""

import os
import sys
import subprocess
import json

# ==========================================
# 1. SYSTEM DEPENDENCY & PACKAGE INSTALLER
# ==========================================

def setup_environment():
    """
    Checks and automatically installs system packages via pkg (APT)
    and PIP libraries required for C++ GGUF inference in Termux.
    """
    print("\n[+] Checking System Environment for Termux & Linux...")
    
    # System package check (Termux pkg)
    if os.path.exists("/data/data/com.termux/files/usr/bin/pkg"):
        print("[+] Termux Environment Detected. Verifying system binaries...")
        system_packages = ["clang", "cmake", "make", "python", "git"]
        for pkg in system_packages:
            try:
                subprocess.run(["pkg", "install", pkg, "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    # Python pip packages required
    required_pip_pkgs = {
        "ctransformers": "ctransformers",
        "rich": "rich",
        "requests": "requests",
        "tqdm": "tqdm",
        "huggingface_hub": "huggingface_hub"
    }

    missing_pkgs = []
    import importlib.util
    for module_name, pip_name in required_pip_pkgs.items():
        if importlib.util.find_spec(module_name) is None:
            missing_pkgs.append(pip_name)

    if missing_pkgs:
        print(f"[!] Missing Python Libraries Found: {missing_pkgs}")
        print("[!] Automatic Installation Starting... Please wait a moment.\n")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_pkgs])
            print("\n[✔] All dependencies installed successfully!\n")
        except Exception as err:
            print(f"\n[❌] Error installing libraries: {err}")
            print("Please run manually: pip install ctransformers rich requests tqdm huggingface_hub")
            sys.exit(1)

# Execute Environment Auto-Fix
setup_environment()

# Import Installed Libraries
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table
from huggingface_hub import hf_hub_download
from ctransformers import AutoModelForCausalLM

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
        "title": "Llama 3.2 1B Instruct (Meta's Latest Small LLM)",
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
    """
    Downloads a model from Hugging Face Hub directly into local storage.
    """
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
    """
    Presents an interactive menu allowing the user to select predefined models
    or enter custom Hugging Face GGUF repository parameters.
    """
    display_header()
    
    table = Table(title="[bold cyan]Select AI Model to Download or Run[/bold cyan]", border_style="cyan")
    table.add_column("Option", style="bold yellow", justify="center")
    table.add_column("Model Name / Details", style="white")
    table.add_column("Approx Size", style="green", justify="center")

    for key, data in PREDEFINED_MODELS.items():
        table.add_row(key, data["title"], data["size"])
    
    table.add_row("C", "Custom HuggingFace GGUF Repo (Enter your own Repo & File)", "Custom")
    
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
        model_type = Prompt.ask("Enter Model Architecture Type (llama / qwen2 / mistral / gemma)", default="llama")
        
        file_path = download_hf_model(repo_id.strip(), filename.strip())
        return file_path, model_type.strip(), filename.strip()
    else:
        selected = PREDEFINED_MODELS[choice]
        file_path = download_hf_model(selected["repo_id"], selected["filename"])
        return file_path, selected["model_type"], selected["filename"]

# ==========================================
# 4. INFERENCE ENGINE & CHAT LOOP
# ==========================================

def initialize_model(model_path, model_type):
    """
    Loads quantized GGUF binary using CTransformers.
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

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type=model_type,
            gpu_layers=0,
            context_length=2048
        )
        return model
    except Exception as error:
        console.print(f"\n[bold red]❌ Failed to initialize CTransformers Model:[/bold red] {error}")
        sys.exit(1)

def run_chat_loop(model, model_filename):
    """
    Executes multi-turn conversation loop with context history and live streaming output.
    """
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

            # Store user message into history
            chat_history.append({"role": "user", "content": cleaned_input})

            # Format full context prompt using ChatML structure
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
                # Fallback to direct completion if streaming fails
                raw_response = model(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                full_response = raw_response.strip()
                console.print(Markdown(full_response))

            # Store AI response into context memory
            chat_history.append({"role": "assistant", "content": full_response.strip()})

        except KeyboardInterrupt:
            console.print("\n\n[bold red]Session Interrupted by User.[/bold red]")
            break
        except Exception as error:
            console.print(f"\n[bold red]⚠️ Generation Error:[/bold red] {error}")

# ==========================================
# 5. ENTRY POINT
# ==========================================

def main():
    model_path, model_type, model_filename = model_selection_menu()
    loaded_model = initialize_model(model_path, model_type)
    run_chat_loop(loaded_model, model_filename)

if __name__ == "__main__":
    main()
    
