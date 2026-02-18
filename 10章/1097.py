import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

train_df = pd.read_csv("./train.tsv", sep="\t")
dev_df   = pd.read_csv("./dev.tsv", sep="\t")

train_texts = train_df["sentence"].tolist()
train_labels = train_df["label"].tolist()

dev_texts = dev_df["sentence"].tolist()
dev_labels = dev_df["label"].tolist()

class SentimentModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)

        return logits

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def encode_texts(texts, labels, max_len=128):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"], torch.tensor(labels)

train_ids, train_mask, train_y = encode_texts(train_texts, train_labels)
dev_ids, dev_mask, dev_y = encode_texts(dev_texts, dev_labels)

train_loader = DataLoader(
    TensorDataset(train_ids, train_mask, train_y),
    batch_size=16,
    shuffle=True
)

dev_loader = DataLoader(
    TensorDataset(dev_ids, dev_mask, dev_y),
    batch_size=16
)

model = SentimentModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total_loss = 0

    for input_ids, mask, labels in train_loader:
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids, mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss / len(train_loader))

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for input_ids, mask, labels in dev_loader:
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, mask)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Dev Accuracy:", correct / total)

def predict(text):
    model.eval()

    ids, mask, _ = encode_texts([text], [0])

    ids = ids.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        logits = model(ids, mask)
        pred = torch.argmax(logits, dim=1)

    return pred.item()
