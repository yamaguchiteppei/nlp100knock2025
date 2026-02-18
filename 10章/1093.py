import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math

# =========================
# モデル
# =========================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# =========================
# PPL計算関数
# =========================
def calc_ppl(sentence):

    encodings = tokenizer(sentence, return_tensors="pt")
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss # 次の単語をどれだけ正しく予測できたかの損失値

    ppl = math.exp(loss.item())
    return ppl

# =========================
# 文
# =========================
sentences = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises"
]

# =========================
# 実行
# =========================
print("=== Perplexity ===")
for s in sentences:
    print(f"{s}")
    print(f"PPL: {calc_ppl(s):.3f}\n")
