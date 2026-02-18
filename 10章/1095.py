import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token   # ★追加

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# =========================
# Multi-turn Prompt
# =========================
prompt = """You are a helpful assistant.

User: What do you call a sweet eaten after dinner?
Assistant: dessert

User: Please give me the plural form of the word with its spelling in reverse order.
Assistant:"""

enc = tokenizer(prompt, return_tensors="pt")
input_ids = enc.input_ids
attention_mask = enc.attention_mask

# =========================
# Generate
# =========================
output = model.generate(
    input_ids,
    attention_mask=attention_mask,   
    max_new_tokens=10,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id  
)

generated = tokenizer.decode(output[0], skip_special_tokens=True)

print("=== Prompt ===")
print(prompt)

print("\n=== Response ===")
print(generated[len(prompt):])
