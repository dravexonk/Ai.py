#!/usr/bin/env python3
"""
Termux Local Hindi AI Assistant
-------------------------------
A lightweight, offline, C++ based Local AI for Termux without PyTorch dependencies.
"""

import sys
import os
try:
    from ctransformers import AutoModelForCausalLM
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.markdown import Markdown
except ImportError as err:
    print(f"\n[!] Missing dependency: {err}")
    print("[!] Please run: pip install -r requirements.txt\n")
    sys.exit(1)

# Rich Console initialization
console = Console()

# Configuration Settings
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
SYSTEM_PROMPT = (
    "आप एक समझदार, उपयोगी और कुशल AI सहायक हैं। "
    "यूज़र के हर सवाल का उत्तर स्पष्ट, सटीक और केवल शुद्ध हिंदी भाषा में दें।"
)


def load_model():
    """Download and initialize the quantized GGUF LLM using CTransformers."""
    console.print(
        Panel.fit(
            "[bold cyan]⏳ AI मॉडल लोड किया जा रहा है...[/bold cyan]\n"
            "[yellow]नोट: पहली बार चलने पर यह ~350MB की मॉडल फाइल ऑटो-डाउनलोड करेगा।[/yellow]",
            title="[bold green]System Initialization[/bold green]",
            border_style="cyan"
        )
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO,
            model_file=MODEL_FILE,
            model_type="qwen2",
            gpu_layers=0,  # Force CPU computation for Mobile Devices
            context_length=1024
        )
        return model
    except Exception as error:
        console.print(f"\n[bold red]❌ मॉडल लोड करने में त्रुटि आई:[/bold red] {error}\n")
        sys.exit(1)


def display_banner():
    """Clears terminal and shows application banner."""
    console.clear()
    banner_text = (
        "[bold magenta]🤖 TERMUX LOCAL HINDI AI (OFFLINE) 🤖[/bold magenta]\n\n"
        "[green]✔ बिना PyTorch और बिना API Key के पूर्णतः सक्रिय[/green]\n"
        "[green]✔ न्यूनतम RAM खपत (Optimized for Android)[/green]\n"
        "[yellow]✔ बाहर निकलने के लिए 'exit', 'quit' या 'बाहर' टाइप करें[/yellow]"
    )
    console.print(
        Panel(
            banner_text,
            title="[bold white]GitHub: Termux AI Core[/bold white]",
            border_style="bright_blue",
            expand=False
        )
    )


def chat_loop(model):
    """Main interactive chat loop."""
    display_banner()
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold bright_blue]आप[/bold bright_blue]")
            
            # Clean and validate input
            cleaned_input = user_input.strip()
            
            if not cleaned_input:
                continue
                
            if cleaned_input.lower() in ["exit", "quit", "exit()", "बाहर", "बाय"]:
                console.print("\n[bold yellow]AI: अलविदा! आपका दिन शुभ हो।[/bold yellow]\n")
                break
            
            # Format prompt with ChatML system tags
            formatted_prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{cleaned_input}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            
            with console.status("[bold green]AI विचार कर रहा है...[/bold green]", spinner="dots"):
                response_raw = model(
                    formatted_prompt,
                    max_new_tokens=300,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                
            response_clean = response_raw.strip()
            
            # Render Markdown output inside styled Panel
            console.print(
                Panel(
                    Markdown(response_clean),
                    title="[bold green]AI का उत्तर[/bold green]",
                    border_style="green"
                )
            )
            
        except KeyboardInterrupt:
            console.print("\n\n[bold red]सेशन जबरदस्ती बंद किया गया।[/bold red]")
            break
        except Exception as error:
            console.print(f"\n[bold red]⚠️ त्रुटि उत्पन्न हुई:[/bold red] {error}")


if __name__ == "__main__":
    ai_model = load_model()
    chat_loop(ai_model)
    
