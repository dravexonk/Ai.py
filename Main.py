#!/usr/bin/env python3
"""
Termux Local Hindi AI Assistant (Advanced Edition)
--------------------------------------------------
A feature-rich, offline, conversational GGUF-based AI assistant for Termux
with multi-turn memory, real-time streaming output, and interactive command support.
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
    print(f"\n[!] आवश्यक डिपेंडेंसी गायब है: {err}")
    print("[!] कृपया रन करें: pip install ctransformers rich\n")
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
            gpu_layers=0,  # Android/Termux के लिए CPU मोड
            context_length=2048  # लंबी बातचीत के लिए कॉन्टेक्स्ट बढ़ाया गया
        )
        return model
    except Exception as error:
        console.print(f"\n[bold red]❌ मॉडल लोड करने में त्रुटि आई:[/bold red] {error}\n")
        sys.exit(1)


def display_banner():
    """Clears terminal and shows application banner with commands info."""
    console.clear()
    banner_text = (
        "[bold magenta]🤖 TERMUX ADVANCED HINDI AI (OFFLINE) 🤖[/bold magenta]\n\n"
        "[green]✔ मल्टी-टर्न चैट मेमोरी (Multi-turn Memory) सक्रिय[/green]\n"
        "[green]✔ लाइव स्ट्रीमिंग आउटपुट (Real-time Token Generation)[/green]\n"
        "[yellow]💡 विशेष कमांड्स:[/yellow]\n"
        "  • [cyan]/clear[/cyan] - चैट इतिहास साफ़ करें\n"
        "  • [cyan]/help[/cyan] - सहायता और निर्देश\n"
        "  • [cyan]/exit[/cyan] या [cyan]बाहर[/cyan] - प्रोग्राम बंद करें"
    )
    console.print(
        Panel(
            banner_text,
            title="[bold white]Termux AI Advanced Core[/bold white]",
            border_style="bright_blue",
            expand=False
        )
    )


def chat_loop(model):
    """Main interactive chat loop with conversation history and streaming."""
    display_banner()
    
    # चैट इतिहास (Memory Buffer) स्टोर करने के लिए लिस्ट
    chat_history = []
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold bright_blue]आप[/bold bright_blue]")
            cleaned_input = user_input.strip()
            
            if not cleaned_input:
                continue
                
            cmd = cleaned_input.lower()
            
            # Exit Commands
            if cmd in ["exit", "quit", "exit()", "बाहर", "बाय", "/exit"]:
                console.print("\n[bold yellow]AI: अलविदा! आपका दिन शुभ हो।[/bold yellow]\n")
                break
            
            # Clear History Command
            elif cmd in ["/clear", "clear", "साफ़"]:
                chat_history.clear()
                console.print("[bold green]✨ चैट मेमोरी पूरी तरह साफ़ कर दी गई है![/bold green]")
                continue
            
            # Help Command
            elif cmd in ["/help", "help", "मदद"]:
                console.print(
                    Panel(
                        "[bold cyan]उपलब्ध कमांड्स की सूची:[/bold cyan]\n"
                        "1. [yellow]/clear[/yellow] - पुरानी बातचीत की याददाश्त मिटाएं।\n"
                        "2. [yellow]/help[/yellow] - यह हेल्प मेनू देखें।\n"
                        "3. [yellow]/exit[/yellow] - असिस्टेंट से बाहर आएं।",
                        title="[bold green]Help Menu[/bold green]",
                        border_style="cyan"
                    )
                )
                continue
            
            # यूज़र के मैसेज को मेमोरी में जोड़ें
            chat_history.append({"role": "user", "content": cleaned_input})
            
            # Qwen ChatML फॉर्मेट के अनुसार पूरा प्रॉम्प्ट तैयार करना (मेमोरी सहित)
            formatted_prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            for message in chat_history:
                role = message["role"]
                content = message["content"]
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            formatted_prompt += "<|im_start|>assistant\n"
            
            console.print("\n[bold green]AI का उत्तर:[/bold green]", end=" ")
            full_response = ""
            
            # लाइव स्ट्रीमिंग जनरेशन (ctransformers stream=True)
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
                print()  # नई लाइन अंत में
                
            except Exception as stream_err:
                console.print(f"\n[bold red]स्ट्रीमिंग त्रुटि:[/bold red] {stream_err}")
                # यदि स्ट्रीमिंग फेल हो जाए तो नॉर्मल जनरेशन का उपयोग करें
                response_raw = model(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    stop=["<|im_end|>"]
                )
                full_response = response_raw.strip()
                console.print(Markdown(full_response))
            
            # AI के जवाब को चैट हिस्ट्री में जोड़ें ताकि आगे का संदर्भ बना रहे
            chat_history.append({"role": "assistant", "content": full_response.strip()})
            
        except KeyboardInterrupt:
            console.print("\n\n[bold red]सेशन जबरदस्ती बंद किया गया।[/bold red]")
            break
        except Exception as error:
            console.print(f"\n[bold red]⚠️ त्रुटि उत्पन्न हुई:[/bold red] {error}")


if __name__ == "__main__":
    ai_model = load_model()
    chat_loop(ai_model)
            
