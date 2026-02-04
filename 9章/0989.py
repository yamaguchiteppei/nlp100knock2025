import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd

# Max Poolingとは？
#複数のベクトル（トークン表現）の中から、
#各次元ごとに最大値だけを取り出して、1つのベクトルにする操作


# =========================
# 設定
# =========================
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# データセット
# =========================
class SST2Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }


# =========================
# Max Pooling 分類モデル
# =========================
class BertMaxPoolClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # (batch, seq_len, hidden)
        hidden = outputs.last_hidden_state

        # padding 部分を無効化
        mask = attention_mask.unsqueeze(-1).expand(hidden.size())
        hidden = hidden.masked_fill(mask == 0, -1e9)

        # max pooling（token方向）
        pooled = torch.max(hidden, dim=1).values

        logits = self.classifier(pooled)
        return logits


# =========================
# データ読み込み
# =========================
def load_sst2(path):
    df = pd.read_csv(path, sep="\t")
    return df["sentence"].tolist(), df["label"].tolist()


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_texts, train_labels = load_sst2("./SST-2/train.tsv")
dev_texts, dev_labels = load_sst2("./SST-2/dev.tsv")

train_ds = SST2Dataset(train_texts, train_labels, tokenizer)
dev_ds = SST2Dataset(dev_texts, dev_labels, tokenizer)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)


# =========================
# 学習
# =========================
model = BertMaxPoolClassifier(MODEL_NAME).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")


# =========================
# 検証（正解率）
# =========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in dev_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f}")
