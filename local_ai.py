import sys
from ctransformers import AutoModelForCausalLM
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

console.print(
    Panel.fit(
        "[bold cyan]⏳ AI मॉडल लोड हो रहा है...[/bold cyan]\n"
        "[yellow]पहली बार चलने पर यह ~350MB का हल्का मॉडल डाउनलोड करेगा।[/yellow]",
        title="[bold green]Termux Direct AI[/bold green]",
        border_style="cyan",
    )
)

try:
  # बिना किसी API Key या PyTorch के सीधे GGUF मॉडल लोड करना
  llm = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
      model_file="qwen2.5-0.5b-instruct-q4_k_m.gguf",
      model_type="qwen2",
      gpu_layers=0,  # CPU पर चलाने के लिए
  )
except Exception as e:
  console.print(
      f"\n[bold red]❌ मॉडल लोड करने में समस्या आई:[/bold red] {e}\n"
  )
  sys.exit(1)

# इंटरफेस रिफ्रेश करें
console.clear()
console.print(
    Panel(
        "[bold magenta]🤖 TERMUX DIRECT HINDI AI 🤖[/bold magenta]\n\n"
        "[green]✔ बिना PyTorch, बिना API Key के सीधे लाइब्रेरी से सक्रिय[/green]\n"
        "[yellow]✔ बाहर निकलने के लिए 'exit' या 'बाहर' टाइप करें[/yellow]",
        title="[bold white]Welcome[/bold white]",
        border_style="bright_blue",
    )
)

# चैट लूप
while True:
  try:
    user_input = Prompt.ask("\n[bold blue]आप[/bold blue]")

    if user_input.strip().lower() in [
        "exit",
        "quit",
        "exit()",
        "बाहर",
        "बाय",
    ]:
      console.print(
          "\n[bold yellow]AI: अलविदा! आपका दिन शुभ हो।[/bold yellow]\n"
      )
      break

    if not user_input.strip():
      continue

    with console.status(
        "[bold green]AI सोच रहा है...[/bold green]", spinner="dots"
    ):
      # एआई के लिए सिस्टम प्रॉम्प्ट
      formatted_prompt = (
          f"<|im_start|>system\nआप एक समझदार और मददगार AI हैं। यूजर के हर"
          f" सवाल का जवाब केवल और केवल शुद्ध हिंदी भाषा में दें।<|im_end|>\n"
          f"<|im_start|>user\n{user_input}<|im_end|>\n"
          f"<|im_start|>assistant\n"
      )

      # डायरेक्ट टेक्स्ट जनरेशन
      response_text = llm(
          formatted_prompt,
          max_new_tokens=250,
          temperature=0.7,
          stop=["<|im_end|>"],
      )

    # आउटपुट डिस्प्ले
    console.print(
        Panel(
            f"[bold white]{response_text.strip()}[/bold white]",
            title="[bold green]AI Response[/bold green]",
            border_style="green",
        )
    )

  except KeyboardInterrupt:
    console.print("\n\n[bold red]सेशन समाप्त किया गया।[/bold red]")
    break
  except Exception as e:
    console.print(f"\n[bold red]⚠️ त्रुटि:[/bold red] {e}")
      
