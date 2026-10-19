#!/usr/bin/env python3
"""
Termux Universal Local AI Engine (Advanced Multi-Model Interactive System)
--------------------------------------------------------------------------
Author: Modern AI Core
Features:
 - Pure CPU Execution for Android / Termux (No PyTorch needed)
 - Self-Healing C++ Compilation Engine for Termux ARM64 Architecture
 - Interactive Hugging Face Downloader & Custom Repo Direct Fetching
 - Multi-Turn Conversation Memory Buffer & Real-Time Token Streaming
 - Comprehensive Error Handling and Automatic NoneType Crash Prevention
"""

import os
import sys
import subprocess
import importlib.util

# ==========================================
# 1. ADVANCED AUTO-ENVIRONMENT SETUP & FIXER
# ==========================================

def run_cmd(command_list, suppress_output=True):
    """Executes shell commands safely with error suppression."""
    try:
        stdout_target = subprocess.DEVNULL if suppress_output else None
        stderr_target = subprocess.DEVNULL if suppress_output else None
        return subprocess.run(command_list, check=True, stdout=stdout_target, stderr=stderr_target)
    except Exception:
        return None

def verify_and_fix_ctransformers():
    """
    Termux missing C++ shared binary error auto-fixer.
    Reinstalls ctransformers from source using Termux system native compilers.
    """
    print("\n[!] CTransformers native binaries missing or compilation mismatch detected.")
    print("[!] Initiating automatic local source compilation (Termux C++ Auto-Fix)...")
    
    if os.path.exists("/data/data/com.termux/files/usr/bin/pkg"):
        run_cmd(["pkg", "install", "clang", "cmake", "make", "ninja", "python", "python-dev", "-y"], suppress_output=False)
    
    print("[!] Compiling ctransformers C++ engine locally for ARM64...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "ctransformers", "-y"])
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ctransformers", "--no-build-isolation", "--no-binary", "ctransformers"
        ])
        print("\n[✔] CTransformers C++ core compiled and installed successfully!\n")
    except Exception as err:
        print(f"\n[❌] Source compilation fallback warning: {err}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "ctransformers", "--prefer-binary"
            ])
        except Exception as e2:
            print(f"\n[❌] Critical installation error: {e2}")
            sys.exit(1)

def setup_environment():
    """
    Checks and automatically installs system packages and Python libraries required for Termux.
    """
    print("\n[+] Checking System Environment for Termux & Linux...[span_0](start_span)[span_0](end_span)")
    
    if os.path.exists("/data/data/com.termux/files/usr/bin/pkg"):
        print("[+] Termux Environment Detected. Ensuring compiler binaries...[span_1](start_span)[span_1](end_span)")
        system_packages = ["clang", "cmake", "make", "ninja", "python", "python-pip", "git", "binutils", "rust"]
        for pkg in system_packages:
            run_cmd(["pkg", "install", pkg, "-y"])

    required_pip_pkgs = [
        "ctransformers",
        "rich",
        "requests",
        "tqdm",
        "huggingface_hub<0.25.0"  # Prevents hf-xet/maturin rust build crashes[span_2](start_span)[span_2](end_span)
    ]

    missing = False
    for mod in ["ctransformers", "rich", "requests", "tqdm", "huggingface_hub"]:
        if importlib.util.find_spec(mod) is None:
            missing = True
            break

    if missing:
        print("[!] Missing Python libraries detected. Installing with Termux compatibility flags...[span_3](start_span)[span_3](end_span)")
        try:
            cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation"] + required_pip_pkgs
            subprocess.check_call(cmd)
            print("\n[✔] Libraries installed successfully[span_4](start_span)[span_4](end_span)!\n")
        except Exception:
            try:
                cmd_fallback = [sys.executable, "-m", "pip", "install", "--prefer-binary"] + required_pip_pkgs
                subprocess.check_call(cmd_fallback)
                print("\n[✔] Libraries installed via fallback flags[span_5](start_span)[span_5](end_span)!\n")
            except Exception as err:
                print(f"\n[❌] Dependency installation error: {err}[span_6](start_span)[span_6](end_span)")
                sys.exit(1)

# Execute Environment Setup
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
        "title": "Qwen 2.5 0.5B Instruct (Ultra Fast / Recommended for Termux)[span_7](start_span)[span_7](end_span)",
        "repo_id": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "model_type": "qwen2",
        "size": "~350 MB[span_8](start_span)[span_8](end_span)"
    },
    "2": {
        "title": "TinyLlama 1.1B Chat (Lightweight & Smart)[span_9](start_span)[span_9](end_span)",
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "model_type": "llama",
        "size": "~630 MB[span_10](start_span)[span_10](end_span)"
    },
    "3": {
        "title": "Qwen 2.5 1.5B Instruct (High Accuracy / Multi-Lingual)[span_11](start_span)[span_11](end_span)",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "model_type": "qwen2",
        "size": "~1.1 GB[span_12](start_span)[span_12](end_span)"
    },
    "4": {
        "title": "Llama 3.2 1B Instruct (Meta's Small LLM)[span_13](start_span)[span_13](end_span)",
        "repo_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "model_type": "llama",
        "size": "~800 MB[span_14](start_span)[span_14](end_span)"
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
        "[green]✔ Zero PyTorch / CPU-Only Native C++ Execution[span_15](start_span)[span_15](end_span)[/green]\n"
        "[green]✔ Interactive HuggingFace Downloader Built-In[span_16](start_span)[span_16](end_span)[/green]\n"
        "[green]✔ Multi-Turn Chat Memory & Streaming Tokens[span_17](start_span)[span_17](end_span)[/green]"
    )
    console.print(
        Panel(
            banner,
            title="[bold white]GitHub Hub: Termux Universal AI[span_18](start_span)[span_18](end_span)[/bold white]",
            border_style="bright_blue",
            expand=False
        )
    )

def download_hf_model(repo_id, filename):
    """Downloads model file from Hugging Face directly into local folder with integrity checking."""
    local_path = os.path.join(MODELS_DIRECTORY, filename)
    
    # Check if file exists and has valid size (> 1MB to prevent corrupted empty downloads)
    if os.path.exists(local_path):
        if os.path.getsize(local_path) > 1024 * 1024:
            console.print(f"\n[bold green]✔ Model file exists locally and is valid: {filename}[/bold green]")
            return local_path
        else:
            console.print(f"\n[bold yellow]⚠️ Existing model file is corrupted or incomplete. Re-downloading...[/bold yellow]")
            os.remove(local_path)

    console.print(f"\n[bold yellow]📥 Downloading from Hugging Face...[span_19](start_span)[span_19](end_span)[/bold yellow]")
    console.print(f"[cyan]Repo ID:[/cyan] {repo_id}")
    console.print(f"[cyan]File Name:[/cyan] {filename}\n")

    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIRECTORY,
            local_dir_use_symlinks=False
        )
        console.print(f"\n[bold green]✔ Download Completed Successfully[span_20](start_span)[span_20](end_span)![/bold green]\n")
        return downloaded_file
    except Exception as error:
        console.print(f"\n[bold red]❌ Download Failed:[span_21](start_span)[span_21](end_span)[/bold red] {error}")
        if os.path.exists(local_path):
            os.remove(local_path)
        sys.exit(1)

def model_selection_menu():
    """Interactive menu for model choices."""
    display_header()
    
    table = Table(title="[bold cyan]Select AI Model to Download or Run[span_22](start_span)[span_22](end_span)[/bold cyan]", border_style="cyan")
    table.add_column("Option", style="bold yellow", justify="center")
    table.add_column("Model Name / Details", style="white")
    table.add_column("Approx Size", style="green", justify="center")

    for key, data in PREDEFINED_MODELS.items():
        table.add_row(key, data["title"], data["size"])
    
    table.add_row("C", "Custom HuggingFace GGUF Repo (Enter Custom Repo & File)", "Custom")
    
    console.print(table)

    choice = Prompt.ask(
        "\n[bold green]Select Option Number or 'C' for Custom[span_23](start_span)[span_23](end_span)[/bold green]",
        choices=list(PREDEFINED_MODELS.keys()) + ["C", "c"],
        default="1"
    )

    if choice.upper() == "C":
        console.print("\n[bold cyan]--- Custom HuggingFace GGUF Setup ---[span_24](start_span)[span_24](end_span)[/bold cyan]")
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
    Loads quantized GGUF model with dynamic error catching, auto-repair, 
    and robust NoneType prevention.
    """
    console.print(
        Panel.fit(
            f"[bold cyan]⏳ Loading GGUF Model into RAM...[span_25](start_span)[span_25](end_span)[/bold cyan]\n"
            f"[yellow]File Path: {model_path}[/yellow]\n"
            f"[green]Thread Optimization: Running on CPU Native Backend[span_26](start_span)[span_26](end_span)[/green]",
            title="[bold green]Model Loader[span_27](start_span)[span_27](end_span)[/bold green]",
            border_style="cyan"
        )
    )

    for attempt in range(2):
        try:
            from ctransformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type=model_type,
                gpu_layers=0,
                context_length=2048
            )
            if model is None:
                raise Exception("Model loader returned None object.")
            return model
        except Exception as error:
            error_msg = str(error)
            if attempt == 0:
                console.print(f"\n[bold yellow]⚠️ Initialization Warning/Binary Error Detected:[/bold yellow] {error_msg}")
                verify_and_fix_ctransformers()
            else:
                console.print(f"\n[bold red]❌ Failed to initialize CTransformers Model:[span_28](start_span)[span_28](end_span)[/bold red] {error_msg}")
                console.print("[yellow]Tip: Your model file might be corrupted. Delete the file inside 'downloaded_models/' and restart.[/yellow]")
                sys.exit(1)
                
    console.print(f"\n[bold red]❌ Critical Error: Model could not be initialized and returned None.[/bold red]")
    sys.exit(1)

def run_chat_loop(model, model_filename):
    """Executes chat session with context buffer and streaming tokens."""
    console.clear()
    console.print(
        Panel(
            f"[bold magenta]🤖 AI SESSION ACTIVE ({model_filename})[span_29](start_span)[span_29](end_span)[/bold magenta]\n\n"
            "[green]Status: Offline Engine Ready[span_30](start_span)[span_30](end_span)[/green]\n"
            "[yellow]Commands:[span_31](start_span)[span_31](end_span)[/yellow]\n"
            "  • [cyan]/clear[/cyan] - Reset Conversation Memory[span_32](start_span)[span_32](end_span)\n"
            "  • [cyan]/help[/cyan]  - Show Available Commands[span_33](start_span)[span_33](end_span)\n"
            "  • [cyan]/exit[/cyan]  - Terminate Program[span_34](start_span)[span_34](end_span)",
            title="[bold white]Interactive Chat[span_35](start_span)[span_35](end_span)[/bold white]",
            border_style="bright_blue",
            expand=False
        )
    )

    chat_history = []

    while True:
        try:
            user_input = Prompt.ask("\n[bold bright_blue]You[span_36](start_span)[span_36](end_span)[/bold bright_blue]")
            cleaned_input = user_input.strip()

            if not cleaned_input:
                continue

            cmd = cleaned_input.lower()

            if cmd in ["exit", "quit", "/exit", "bye"]:
                console.print("\n[bold yellow]AI: Goodbye! Have a great time.[span_37](start_span)[span_37](end_span)[/bold yellow]\n")
                break

            elif cmd in ["/clear", "clear"]:
                chat_history.clear()
                console.print("[bold green]✨ Conversation history cleared successfully[span_38](start_span)[span_38](end_span)![/bold green]")
                continue

            elif cmd in ["/help", "help"]:
                console.print(
                    Panel(
                        "[bold cyan]Control Commands:[span_39](start_span)[span_39](end_span)[/bold cyan]\n"
                        " 1. [yellow]/clear[/yellow] - Erase context memory.[span_40](start_span)[span_40](end_span)\n"
                        " 2. [yellow]/help[/yellow]  - Show menu details.[span_41](start_span)[span_41](end_span)\n"
                        " 3. [yellow]/exit[/yellow]  - Exit the program.[span_42](start_span)[span_42](end_span)",
                        title="Help Options[span_43](start_span)[span_43](end_span)",
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

            console.print("\n[bold green]AI Response:[span_44](start_span)[span_44](end_span)[/bold green]", end=" ")
            full_response = ""

            # Real-Time Token Generation via Streaming with Safety Checks
            try:
                if model is None:
                    raise Exception("Model instance is None during generation call.")
                
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

            except Exception as stream_err:
                try:
                    raw_response = model(
                        formatted_prompt,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9,
                        stop=["<|im_end|>"]
                    )
                    full_response = raw_response.strip()
                    console.print(Markdown(full_response))
                except Exception as fallback_err:
                    console.print(f"\n[bold red]⚠️ Generation Error:[span_45](start_span)[span_45](end_span)[/bold red] {fallback_err}")
                    continue

            # Save AI response into history
            if full_response.strip():
                chat_history.append({"role": "assistant", "content": full_response.strip()})

        except KeyboardInterrupt:
            console.print("\n\n[bold red]Session Interrupted by User.[span_46](start_span)[span_46](end_span)[/bold red]")
            break
        except Exception as error:
            console.print(f"\n[bold red]⚠️ Runtime Error:[span_47](start_span)[span_47](end_span)[/bold red] {error}")

# ==========================================
# 5. MAIN EXECUTION ENTRY POINT
# ==========================================

def main():
    model_path, model_type, model_filename = model_selection_menu()
    loaded_model = initialize_model(model_path, model_type)
    run_chat_loop(loaded_model, model_filename)

if __name__ == "__main__":
    main()
 
