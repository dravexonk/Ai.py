#!/usr/bin/env python3
"""
Termux Universal Local AI Engine (Production-Ready Self-Healing Core)
--------------------------------------------------------------------------
Author: Modern AI Core
Features:
 - Pure CPU Execution for Android / Termux (No PyTorch needed)[span_0](start_span)[span_0](end_span)
 - Automatic Self-Healing C++ Compilation Engine (Fixes NoneType & Binary Errors)
 - Interactive Hugging Face Downloader & Custom Repo Direct Fetching[span_1](start_span)[span_1](end_span)
 - Multi-Turn Conversation Memory Buffer & Real-Time Token Streaming[span_2](start_span)[span_2](end_span)
 - Robust Null-Safety Validation to Prevent Runtime Crashes
"""

import os
import sys
import subprocess
import importlib.util

# ==========================================
# 1. ADVANCED AUTOMATED ENVIRONMENT SETUP
# ==========================================

def run_quiet_cmd(cmd_list):
    """Executes a system command silently without throwing uncaught exceptions."""
    try:
        subprocess.run(cmd_list, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def verify_and_rebuild_ctransformers():
    """
    Self-healing routine to compile ctransformers locally from source 
    in Termux/Linux environments when precompiled binaries fail.
    """
    print("\n[!] CTransformers native binaries missing or incompatible.")
    print("[!] Initiating automatic source compilation for ARM64/Android...")
    
    if os.path.exists("/data/data/com.termux/files/usr/bin/pkg"):
        run_quiet_cmd(["pkg", "install", "clang", "cmake", "make", "ninja", "python", "python-dev", "-y"])
    
    try:
        print("[!] Rebuilding ctransformers locally (this may take a minute)...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "ctransformers", "-y"])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ctransformers", "--no-build-isolation", "--no-binary", "ctransformers"
        ])
        print("\n[✔] CTransformers C++ backend compiled and reinstalled successfully!\n")
    except Exception as err:
        print(f"\n[❌] Automatic source compilation failed: {err}")
        print("[!] Please execute manually in Termux:")
        print("    pkg install clang cmake make ninja -y")
        print("    pip install ctransformers --no-build-isolation --no-binary ctransformers")
        sys.exit(1)

def setup_environment():
    """
    Checks and automatically installs system packages via pkg (APT)
    and specific wheel-compatible PIP libraries for Termux[span_3](start_span)[span_3](end_span).
    """
    print("\n[+] Checking System Environment for Termux & Linux...[span_4](start_span)[span_4](end_span)")
    
    if os.path.exists("/data/data/com.termux/files/usr/bin/pkg"):
        print("[+] Termux Environment Detected. Installing required compiler binaries...[span_5](start_span)[span_5](end_span)")
        system_packages = ["clang", "cmake", "make", "ninja", "python", "python-pip", "git", "binutils", "rust"]
        for pkg in system_packages:
            run_quiet_cmd(["pkg", "install", pkg, "-y"])

    required_pip_pkgs = [
        "ctransformers",
        "rich",
        "requests",
        "tqdm",
        "huggingface_hub<0.25.0"  # Avoids hf-xet maturin Rust build errors in Termux[span_6](start_span)[span_6](end_span)
    ]

    missing = False
    for mod in ["ctransformers", "rich", "requests", "tqdm", "huggingface_hub"]:
        if importlib.util.find_spec(mod) is None:
            missing = True
            break

    if missing:
        print("[!] Missing or incompatible Python libraries found.")
        print("[!] Starting auto-installation with Termux compatibility flags...\n")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation"] + required_pip_pkgs
            subprocess.check_call(cmd)
            print("\n[✔] All dependencies installed successfully!\n")
        except Exception:
            try:
                cmd_fallback = [sys.executable, "-m", "pip", "install", "--prefer-binary"] + required_pip_pkgs
                subprocess.check_call(cmd_fallback)
                print("\n[✔] All dependencies installed successfully via fallback!\n")
            except Exception as err2:
                print(f"\n[❌] Error installing libraries: {err2}")
                sys.exit(1)

# Execute Environment Auto-Fix
setup_environment()

# Import Installed Libraries Safely
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
    """Displays terminal ASCII Header Banner[span_7](start_span)[span_7](end_span)."""
    console.clear()
    banner = (
        "[bold magenta]🤖 TERMUX LOCAL AI ENGINE (OFFLINE CORE) 🤖[/bold magenta]\n\n"
        "[green]✔ Zero PyTorch / CPU-Only Native C++ Execution[span_8](start_span)[span_8](end_span)[/green]\n"
        "[green]✔ Interactive HuggingFace Downloader Built-In[span_9](start_span)[span_9](end_span)[/green]\n"
        "[green]✔ Multi-Turn Chat Memory & Streaming Tokens[span_10](start_span)[span_10](end_span)[/green]"
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
    Downloads a model from Hugging Face Hub directly into local storage[span_11](start_span)[span_11](end_span).
    Includes validation to prevent corrupt file initialization.
    """
    local_path = os.path.join(MODELS_DIRECTORY, filename)
    if os.path.exists(local_path):
        # Verify file size sanity (GGUF files should be greater than 1MB)
        if os.path.getsize(local_path) > 1024 * 1024:
            console.print(f"\n[bold green]✔ Model file exists locally: {filename}[/bold green][span_12](start_span)[span_12](end_span)")
            return local_path
        else:
            console.print(f"\n[bold yellow]⚠️ Existing model file is corrupt or incomplete. Re-downloading...[/bold yellow]")
            os.remove(local_path)

    console.print(f"\n[bold yellow]📥 Downloading from Hugging Face...[/bold yellow][span_13](start_span)[span_13](end_span)")
    console.print(f"[cyan]Repo ID:[/cyan] {repo_id}[span_14](start_span)[span_14](end_span)")
    console.print(f"[cyan]File Name:[/cyan] {filename}\n")[span_15](start_span)[span_15](end_span)

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIRECTORY,
            local_dir_use_symlinks=False
        )
        console.print(f"\n[bold green]✔ Download Completed Successfully![/bold green]\n")[span_16](start_span)[span_16](end_span)
        return downloaded_file
    except Exception as error:
        console.print(f"\n[bold red]❌ Download Failed:[/bold red] {error}[span_17](start_span)[span_17](end_span)")
        if os.path.exists(local_path):
            os.remove(local_path)
        sys.exit(1)

def model_selection_menu():
    """
    Presents an interactive menu allowing the user to select predefined models
    or enter custom Hugging Face GGUF repository parameters[span_18](start_span)[span_18](end_span).
    """
    display_header()
    
    table = Table(title="[bold cyan]Select AI Model to Download or Run[/bold cyan]", border_style="cyan")
    table.add_column("Option", style="bold yellow", justify="center")
    table.add_column("Model Name / Details", style="white")
    table.add_column("Approx Size", style="green", justify="center")

    for key, data in PREDEFINED_MODELS.items():
        table.add_row(key, data["title"], data["size"])
    
    table.add_row("C", "Custom HuggingFace GGUF Repo (Enter your own Repo & File)", "Custom")[span_19](start_span)[span_19](end_span)
    
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
# 4. INFERENCE ENGINE & NULL-SAFE INITIALIZER
# ==========================================

def initialize_model(model_path, model_type):
    """
    Loads quantized GGUF binary using CTransformers[span_20](start_span)[span_20](end_span) with built-in 
    self-healing recovery and strict null-safety validation to stop NoneType errors.
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

    for attempt in range(2):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                gpu_layers=0,
                context_length=2048
            )
            if model is None:
                raise ValueError("Model object returned None.")
            return model
        except Exception as error:
            error_msg = str(error)
            if attempt == 0 and ("Precompiled binaries" in error_msg or "DLL load failed" in error_msg or "None" in error_msg):
                console.print(f"\n[bold yellow]⚠️ Binary or initialization error detected. Triggering self-fix...[/bold yellow]")
                verify_and_rebuild_ctransformers()
            else:
                console.print(f"\n[bold red]❌ Failed to initialize CTransformers Model:[/bold red] {error}[span_21](start_span)[span_21](end_span)")
                console.print("[yellow]Hint: Your model file might be corrupted. Delete the 'downloaded_models/' folder and restart.[/yellow]")
                sys.exit(1)

    console.print(f"\n[bold red]❌ Critical Error: Model could not be initialized.[/bold red]")
    sys.exit(1)

def run_chat_loop(model, model_filename):
    """
    Executes multi-turn conversation loop with context history and live streaming output[span_22](start_span)[span_22](end_span).
    Includes guards against NoneType callable exceptions.
    """
    if model is None:
        console.print("\n[bold red]❌ Error: Model instance is None. Cannot start chat loop.[/bold red]")
        sys.exit(1)

    console.clear()
    console.print(
        Panel(
            f"[bold magenta]🤖 AI SESSION ACTIVE ({model_filename})[/bold magenta]\n\n"
            "[green]Status: Offline Engine Ready[/green]\n"
            "[yellow]Commands:[/yellow]\n"
            "  • [cyan]/clear[/cyan] - Reset Conversation Memory[span_23](start_span)[span_23](end_span)\n"
            "  • [cyan]/help[/cyan]  - Show Available Commands[span_24](start_span)[span_24](end_span)\n"
            "  • [cyan]/exit[/cyan]  - Terminate Program[span_25](start_span)[span_25](end_span)",
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

            # Real-Time Token Generation via Streaming[span_26](start_span)[span_26](end_span)
            try:
                stream = model(
                    formatted_prompt,
                    stream=True,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                if stream:
                    for token in stream:
                        full_response += token
                        print(token, end="", flush=True)
                    print()
                else:
                    raise Exception("Stream is empty.")

            except Exception:
                # Fallback to direct completion if streaming fails[span_27](start_span)[span_27](end_span)
                raw_response = model(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                if raw_response:
                    full_response = raw_response.strip()
                    console.print(Markdown(full_response))
                else:
                    full_response = "Error: Failed to generate response."
                    console.print(f"[bold red]{full_response}[/bold red]")

            # Store AI response into context memory[span_28](start_span)[span_28](end_span)
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
 
