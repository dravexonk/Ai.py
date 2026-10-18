import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("=" * 50)
print("⏳ लोकल AI मॉडल लोड हो रहा है...")
print("=" * 50)

# सबसे हल्का और तेज़ मॉडल जो हिंदी समझता है
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32, device_map="cpu"
)

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200
)

print("\n🤖 Termux Local AI तैयार है!")
print("बाहर निकलने के लिए 'exit' या 'बाहर' टाइप करें।\n" + "=" * 50)

while True:
  user_input = input("आप: ")

  if user_input.lower() in ["exit", "quit", "बाहर"]:
    print("\nAI: अलविदा!")
    break

  if not user_input.strip():
    continue

  try:
    messages = [
        {
            "role": "system",
            "content": (
                "आप एक मददगार AI हैं। यूजर के हर सवाल का जवाब केवल और केवल शुद्ध"
                " हिंदी भाषा में दें।"
            ),
        },
        {"role": "user", "content": user_input},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_p=0.9)
    response_text = outputs[0]["generated_text"][len(prompt) :].strip()

    print(f"\nAI: {response_text}\n" + "-" * 40)

  except Exception as e:
    print(f"\n[!] त्रुटि: {e}\n")
    
