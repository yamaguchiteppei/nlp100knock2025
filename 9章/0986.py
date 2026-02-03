import torch
from transformers import AutoTokenizer
import pandas as pd

# ===== データ読み込み =====
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=0)
    return df["sentence"].tolist(), df["label"].tolist()

train_path = "./SST-2/train.tsv"
train_texts, train_labels = load_data(train_path)

# 冒頭4事例を使用
batch_texts = train_texts[:4]
batch_labels = train_labels[:4]

# ===== トークナイザー =====
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# ===== トークン化 + padding =====
batch_inputs = tokenizer(
    batch_texts,
    padding=True,          # 長さを揃える
    truncation=True,       # 長すぎる文は切る
    return_tensors="pt"    # PyTorch Tensor
)

# ===== 結果 =====
print("input_ids shape:", batch_inputs["input_ids"].shape)
print("attention_mask shape:", batch_inputs["attention_mask"].shape)
print("labels:", batch_labels)
