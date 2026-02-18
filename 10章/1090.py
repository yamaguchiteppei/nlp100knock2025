import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# =========================
# ① モデル読み込み（Decoder-Only）
# =========================
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# =========================
# ② 入力文
# =========================
text = "The movie was full of"

# --- トークン化確認 ---
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("=== Tokenization ===")
print("Tokens :", tokens)
print("Token IDs :", token_ids)
print()

# =========================
# ③ モデル入力
# =========================
input_ids = torch.tensor([token_ids])

# =========================
# ④ 次トークン確率計算
# =========================
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1]  # 最後の位置 → 次トークン用

# softmaxで確率化
probs = torch.softmax(logits, dim=0)

# =========================
# ⑤ 上位10トークン取得
# =========================
topk = torch.topk(probs, 10)

top_tokens = topk.indices
top_probs = topk.values

print("=== Top 10 Next Tokens ===")
for i, (tok_id, prob) in enumerate(zip(top_tokens, top_probs), 1):
    token_str = tokenizer.decode([tok_id])
    print(f"{i:2d}: Token: '{token_str}'  |  Probability: {prob.item():.6f}")
