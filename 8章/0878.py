import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# collate 関数（パディング）
def collate(batch):
    batch = sorted(batch, key=lambda x: len(x["input_ids"]), reverse=True)
    max_len = len(batch[0]["input_ids"])

    input_ids = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
        input_ids.append(ids)

    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.stack([item["label"] for item in batch])
    }


# Dataset
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# モデル（Embeddingを更新＝fine-tuning）
class FineTuningClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)   # (B, T, D)
        pooled = emb.mean(dim=1)           # (B, D)
        return self.fc(pooled)             # (B, 1)


# main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- ダミーデータ（実際は SST-2 等を使用）----
    train_data = [
        {"input_ids": torch.tensor([12, 53, 89, 33]), "label": torch.tensor([1.])},
        {"input_ids": torch.tensor([91, 7, 44]), "label": torch.tensor([0.])},
        {"input_ids": torch.tensor([5, 18, 72, 60, 9]), "label": torch.tensor([1.])},
    ]

    dev_data = train_data

    train_loader = DataLoader(
        TextDataset(train_data),
        batch_size=2,
        shuffle=True,
        collate_fn=collate
    )

    dev_loader = DataLoader(
        TextDataset(dev_data),
        batch_size=2,
        shuffle=False,
        collate_fn=collate
    )

    # ---- モデル ----
    model = FineTuningClassifier(
        vocab_size=100000,
        embed_dim=128
    ).to(device)

    #  ファインチューニング
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # 学習
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # 開発セット評価（正解率）
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dev_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids)
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.numel()

    acc = correct / total
    print("開発セット正解率:", acc)


# 実行
if __name__ == "__main__":
    main()
