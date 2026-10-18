#!/usr/bin/env python3
"""
Termux Local Hindi AI Assistant (Universal GGUF Runner)
-----------------------------------------------------
GitHub Repository: Termux AI Core
Features: Auto-Dependency Installer, Interactive Model Menu, Auto-Download, Memory & Streaming.
"""

import os
import sys
import subprocess

def auto_install_dependencies():
    """टर्मक्स में बिना किसी झंझट के ज़रूरी पैकेज खुद इंस्टॉल करना"""
    required_packages = {"ctransformers", "rich", "requests", "tqdm"}
    import importlib.util
    
    missing = [pkg for pkg in required_packages if importlib.util.find_spec(pkg) is None]
    if missing:
        print(f"\n[!] ज़रूरी लाइब्रेरीज गायब हैं: {missing}")
        print("[!] कृपया प्रतीक्षा करें, ऑटोमैटिक सेटअप चल रहा है...\n")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("\n[✔] सभी पैकेज सफलतापूर्वक इंस्टॉल हो गए हैं!\n")
        except Exception as e:
            print(f"\n[❌] डिपेंडेंसी इंस्टॉल करने में विफल: {e}")
            print("मैन्युअल रूप से रन करें: pip install ctransformers rich requests tqdm")
            sys.exit(1)

# पैकेज चेक और ऑटो-इंस्टॉल फंक्शन कॉल
auto_install_dependencies()

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
import requests
from tqdm import tqdm
from ctransformers import AutoModelForCausalLM

console = Console()

# उपलब्ध मॉडल्स की सूची (आप अपनी पसंद के अनुसार जोड़ या बदल सकते हैं)
AVAILABLE_MODELS = {
    "1": {
        "name": "Qwen2.5-0.5B-Instruct (बेस्ट और सबसे तेज़ - ~350MB)",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "type": "qwen2",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    },
    "2": {
        "name": "TinyLlama-1.1B-Chat (हल्का और स्मार्ट - ~630MB)",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "type": "llama",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    },
    "3": {
        "name": "Qwen2.5-1.5B-Instruct (हाई क्वालिटी हिंदी - ~1.1GB)",
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "type": "qwen2",
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    }
}

SYSTEM_PROMPT = (
    "आप एक समझदार, उपयोगी और कुशल AI सहायक हैं। "
    "यूज़र के हर सवाल का उत्तर स्पष्ट, सटीक और केवल शुद्ध हिंदी भाषा में दें।"
)


def download_model_if_not_exists(filename, url):
    """अगर मॉडल फाइल मौजूद नहीं है, तो प्रोग्रेस बार के साथ डाउनलोड करें"""
    if os.path.exists(filename):
        return filename
    
    console.print(f"\n[bold yellow]📥 मॉडल फाइल लोकल डायरेक्टरी में नहीं मिली। डाउनलोड शुरू हो रहा है...[/bold yellow]")
    console.print(f"[cyan]यूआरएल: {url}[/cyan]\n")
    
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
                
        console.print(f"\n[bold green]✔ मॉडल सफलतापूर्वक डाउनलोड हो गया![/bold green]\n")
        return filename
    except Exception as e:
        console.print(f"\n[bold red]❌ डाउनलोड विफल रहा:[/bold red] {e}")
        if os.path.exists(filename):
            os.remove(filename)
        sys.exit(1)


def select_model_menu():
    """यूज़र को मॉडल चुनने के लिए नंबर-वाइज मेनू दिखाना"""
    console.clear()
    banner = (
        "[bold magenta]🤖 TERMUX UNIVERSAL HINDI AI SERVER 🤖[/bold magenta]\n\n"
        "[green]✔ ऑटो-डिपेंडेंसी इंस्टॉलेशन सक्रिय[/green]\n"
        "[green]✔ ऑफलाइन GGUF मॉडल सपोर्ट[/green]"
    )
    console.print(Panel(banner, title="[bold white]GitHub Main Core[/bold white]", border_style="bright_blue", expand=False))
    
    console.print("\n[bold cyan]कृपया उपयोग करने के लिए एक AI मॉडल चुनें:[/bold cyan]\n")
    for key, info in AVAILABLE_MODELS.items():
        console.print(f"  [bold yellow][{key}][/bold yellow] {info['name']}")
    
    choice = Prompt.ask("\n[bold green]मॉडल का नंबर दर्ज करें (जैसे: 1)[/bold green]", choices=list(AVAILABLE_MODELS.keys()), default="1")
    
    selected = AVAILABLE_MODELS[choice]
    return selected


def main():
    # 1. मॉडल चुनें
    model_info = select_model_menu()
    
    # 2. फाइल चेक और डाउनलोड करें
    download_model_if_not_exists(model_info["file"], model_info["url"])
    
    # 3. मॉडल लोड करें
    console.print(
        Panel.fit(
            f"[bold cyan]⏳ AI मॉडल ({model_info['file']}) लोड किया जा रहा है...[/bold cyan]\n"
            "[yellow]नोट: टर्मक्स सीपीयू मोड (Mobile Optimized) का उपयोग हो रहा है।[/yellow]",
            title="[bold green]System Initialization[/bold green]",
            border_style="cyan"
        )
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_info["file"],
            model_type=model_info["type"],
            gpu_layers=0,
            context_length=2048
        )
    except Exception as error:
        console.print(f"\n[bold red]❌ मॉडल लोड करने में त्रुटि:[/bold red] {error}\n")
        sys.exit(1)
        
    # 4. चैट लूप शुरू करें
    console.clear()
    console.print(
        Panel(
            f"[bold magenta]🤖 AI चैट सत्र शुरू हो गया है! ({model_info['name']})[/bold magenta]\n\n"
            "[yellow]💡 विशेष कमांड्स:[/yellow]\n"
            "  • [cyan]/clear[/cyan] - चैट मेमोरी साफ़ करें\n"
            "  • [cyan]/help[/cyan] - सहायता मेनू\n"
            "  • [cyan]/exit[/cyan] या [cyan]बाहर[/cyan] - बंद करें",
            title="[bold white]Termux AI Ready[/bold white]",
            border_style="bright_blue",
            expand=False
        )
    )
    
    chat_history = []
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold bright_blue]आप[/bold bright_blue]")
            cleaned = user_input.strip()
            
            if not cleaned:
                continue
                
            cmd = cleaned.lower()
            if cmd in ["exit", "quit", "बाहर", "बाय", "/exit"]:
                console.print("\n[bold yellow]AI: अलविदा! आपका दिन शुभ हो।[/bold yellow]\n")
                break
            elif cmd in ["/clear", "clear", "साफ़"]:
                chat_history.clear()
                console.print("[bold green]✨ चैट इतिहास साफ़ कर दिया गया है![/bold green]")
                continue
            elif cmd in ["/help", "help", "मदद"]:
                console.print(Panel("[cyan]/clear[/cyan] - मेमोरी रीसेट करें\n[cyan]/exit[/cyan] - बाहर निकलें", title="Help", border_style="cyan"))
                continue
                
            chat_history.append({"role": "user", "content": cleaned})
            
            # प्रॉम्प्ट तैयार करना
            formatted_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            for msg in chat_history:
                formatted_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            formatted_prompt += "<|im_start|>assistant\n"
            
            console.print("\n[bold green]AI का उत्तर:[/bold green]", end=" ")
            full_response = ""
            
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
                response_raw = model(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                full_response = response_raw.strip()
                console.print(Markdown(full_response))
                
            chat_history.append({"role": "assistant", "content": full_response.strip()})
            
        except KeyboardInterrupt:
            console.print("\n\n[bold red]सेशन बंद कर दिया गया।[/bold red]")
            break
        except Exception as err:
            console.print(f"\n[bold red]⚠️ त्रुटि:[/bold red] {err}")


if __name__ == "__main__":
    main()
    
