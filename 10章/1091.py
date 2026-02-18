import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# =========================
# モデル
# =========================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = "The movie was full of"

# ★ tokenizerの正しい使い方（attention_mask取得）
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# =========================
# 生成関数
# =========================
def generate_text(decode_method, temperature=1.0, top_k=50, top_p=0.9):

    if decode_method == "greedy":#常に確率が最も高いトークンを選択
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=20,
            do_sample=False, #常に確率が最も高いトークンを選択
            pad_token_id=tokenizer.eos_token_id
        )

    elif decode_method == "temperature": #確率に応じてランダムに選ぶ方法
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=20,
            do_sample=True, #確率に応じてランダムに選ぶ
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    elif decode_method == "top_k": #「あり得る単語だけで創造性を出す」生成方法
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=20,
            do_sample=True,
            top_k=top_k,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    elif decode_method == "top_p": #「確率の高い単語だけで創造性を出す」生成方法
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=20,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# =========================
# 実験
# =========================
print("=== Greedy ===")
print(generate_text("greedy"))

print("\n=== Temperature 0.5 ===")
for _ in range(3):
    print(generate_text("temperature", temperature=0.5))

print("\n=== Temperature 1.0 ===")
for _ in range(3):
    print(generate_text("temperature", temperature=1.0))

print("\n=== Temperature 1.5 ===")
for _ in range(3):
    print(generate_text("temperature", temperature=1.5))

print("\n=== Top-k Sampling ===")
for _ in range(3):
    print(generate_text("top_k", temperature=1.0, top_k=50))

print("\n=== Top-p Sampling ===")
for _ in range(3):
    print(generate_text("top_p", temperature=1.0, top_p=0.9))
