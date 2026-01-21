import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# collate 関数（パディング）
def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
        input_ids.append(ids)

    return {
        "input_ids": torch.stack(input_ids),
        "label": torch.stack([x["label"] for x in batch])
    }


# Dataset
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# CNNモデル（TextCNN）
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )

        # カーネルサイズ 3,4,5 の畳み込み
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=3),
            nn.Conv1d(embed_dim, num_filters, kernel_size=4),
            nn.Conv1d(embed_dim, num_filters, kernel_size=5),
        ])

        self.fc = nn.Linear(num_filters * 3, 1)

    def forward(self, input_ids):
        # input_ids: (B, T)
        x = self.embedding(input_ids)      # (B, T, D)
        x = x.transpose(1, 2)              # (B, D, T)

        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))        # (B, F, T')
            p = torch.max(c, dim=2)[0]     # (B, F)
            conv_outs.append(p)

        x = torch.cat(conv_outs, dim=1)    # (B, F*3)
        return self.fc(x)                  # (B, 1)

# main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- ダミーデータ（本来は SST-2）----
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
    model = TextCNN(
        vocab_size=100000,
        embed_dim=128,
        num_filters=100
    ).to(device)

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
