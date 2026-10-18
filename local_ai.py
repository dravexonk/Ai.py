import sys
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# कलरफुल आउटपुट के लिए Console इनिशियलाइज करें
console = Console()

# CPU पर थ्रेड्स लिमिट करें ताकि टर्मक्स क्रैश न हो
torch.set_num_threads(2)

console.print(
    Panel.fit(
        "[bold cyan]⏳ Termux Local AI मॉडल लोड हो रहा है...\n[yellow]कृपया"
        " थोड़ा धैर्य रखें, पहली बार में कुछ सेकंड लग सकते हैं।[/yellow]",
        title="[bold green]System Status[/bold green]",
        border_style="cyan",
    )
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

try:
  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
  model = AutoModelForCausalLM.from_pretrained(
      MODEL_ID,
      torch_dtype=torch.float32,
      device_map="cpu",
      low_cpu_mem_usage=True,
  )

  pipe = pipeline(
      "text-generation",
      model=model,
      tokenizer=tokenizer,
      max_new_tokens=250,
      do_sample=True,
      temperature=0.7,
      top_p=0.9,
  )

except Exception as err:
  console.print(
      f"\n[bold red]❌ मॉडल लोड करने में एरर आया:[/bold red] {err}\n"
  )
  sys.exit(1)

# ऐप की सुंदर Header स्क्रीन
console.clear()
console.print(
    Panel(
        "[bold magenta]🤖 TERMUX LOCAL HINDI AI 🤖[/bold magenta]\n\n"
        "[green]✔ बिना API Key के सीधे आपके फोन पर एक्टिव[/green]\n"
        "[yellow]✔ चैट बंद करने के लिए 'exit' या 'exit()' टाइप करें[/yellow]",
        title="[bold white]Welcome[/bold white]",
        border_style="bright_blue",
        expand=False,
    )
)

# चैट लूप
while True:
  try:
    user_input = Prompt.ask("\n[bold bold_blue]आप[/bold bold_blue]")

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
      messages = [
          {
              "role": "system",
              "content": (
                  "आप एक समझदार AI हैं। यूजर के हर सवाल का जवाब स्पष्ट और"
                  " केवल हिंदी भाषा में दें।"
              ),
          },
          {"role": "user", "content": user_input},
      ]

      prompt = tokenizer.apply_chat_template(
          messages, tokenize=False, add_generation_prompt=True
      )
      raw_output = pipe(prompt)
      response_text = raw_output[0]["generated_text"][len(prompt) :].strip()

    # AI का रिस्पॉन्स बॉक्स के अंदर कलरफुल तरीके से दिखाएं
    console.print(
        Panel(
            f"[bold white]{response_text}[/bold white]",
            title="[bold green]AI Response[/bold green]",
            border_style="green",
        )
    )

  except KeyboardInterrupt:
    console.print("\n\n[bold red]सेशन समाप्त किया गया।[/bold red]")
    break
  except Exception as e:
    console.print(f"\n[bold red]⚠️ एक एरर आया:[/bold red] {e}")
      
