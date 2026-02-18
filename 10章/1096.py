import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# =========================
# モデル
# =========================
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# =========================
# SST-2 読み込み
# =========================
df = pd.read_csv("./dev.tsv", sep="\t")

texts = df["sentence"].tolist()
labels = df["label"].tolist()

# =========================
# 推論関数
# =========================
def predict_sentiment(text):

    prompt = f"Review: {text}\nSentiment:"

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(output[0])
    answer = generated[len(prompt):].strip().lower()

    if "positive" in answer:
        return 1
    elif "negative" in answer:
        return 0
    else:
        return None


# =========================
# 評価
# =========================
correct = 0
total = 0

for text, label in tqdm(zip(texts, labels), total=len(texts)):
    pred = predict_sentiment(text)
    if pred is not None:
        total += 1
        if pred == label:
            correct += 1

accuracy = correct / total

print("Accuracy:", accuracy)
