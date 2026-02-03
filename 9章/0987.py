import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score


# =========================
# 設定
# =========================
MODEL_ID = "answerdotai/ModernBERT-base"
TRAIN_PATH = "./SST-2/train.tsv"
DEV_PATH = "./SST-2/dev.tsv"
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128


# =========================
# データ読み込み
# =========================
def load_data(path):
    df = pd.read_csv(path, sep="\t")
    return df["sentence"].tolist(), df["label"].tolist()


train_texts, train_labels = load_data(TRAIN_PATH)
dev_texts, dev_labels = load_data(DEV_PATH)


# =========================
# トークナイザ & モデル
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# =========================
# トークナイズ
# =========================
def tokenize(texts, labels):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    labels = torch.tensor(labels)
    return enc, labels


train_enc, train_labels = tokenize(train_texts, train_labels)
dev_enc, dev_labels = tokenize(dev_texts, dev_labels)


# =========================
# DataLoader
# =========================
train_dataset = TensorDataset(
    train_enc["input_ids"],
    train_enc["attention_mask"],
    train_labels
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# =========================
# ファインチューニング
# =========================
optimizer = AdamW(model.parameters(), lr=LR)

model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0

    for input_ids, attention_mask, labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")


# =========================
# 検証（正解率）
# =========================
model.eval()
preds = []
gold = []

with torch.no_grad():
    for i in range(0, len(dev_texts), BATCH_SIZE):
        batch_texts = dev_texts[i:i+BATCH_SIZE]
        batch_labels = dev_labels[i:i+BATCH_SIZE]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        ).to(device)

        outputs = model(**enc)
        predictions = torch.argmax(outputs.logits, dim=1)

        preds.extend(predictions.cpu().tolist())
        gold.extend(batch_labels)


acc = accuracy_score(gold, preds)
print(f"\nValidation Accuracy: {acc:.4f}")

model.save_pretrained("finetuned-sst2")
tokenizer.save_pretrained("finetuned-sst2")
