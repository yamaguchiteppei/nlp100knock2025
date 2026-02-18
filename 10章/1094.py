import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# =========================
# Chat Template
# =========================

prompt = """You are a helpful assistant.

User: What do you call a sweet eaten after dinner?
Assistant:"""

input_ids = tokenizer.encode(prompt, return_tensors="pt")

# =========================
# 生成
# =========================
output = model.generate(
    input_ids,
    max_new_tokens=30,
    temperature=0.7,
    do_sample=True
)

generated = tokenizer.decode(output[0], skip_special_tokens=True)

print("=== Prompt ===")
print(prompt)

print("\n=== Response ===")
print(generated[len(prompt):])

