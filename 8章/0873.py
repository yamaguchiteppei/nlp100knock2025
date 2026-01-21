import torch
import torch.nn as nn
import pandas as pd
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader

# データ読み込み
def load_sst2(path):
    return pd.read_csv(path, sep="\t")

# 語彙構築
def build_vocab(*dfs):
    vocab = set()
    for df in dfs:
        for s in df["sentence"]:
            vocab.update(s.lower().split())
    return vocab

# モデル
class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 単語埋め込み
def load_embeddings(path, vocab):
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)

    word2id = {"<PAD>": 0}
    vectors = [torch.zeros(w2v.vector_size)]

    for w in vocab:
        if w in w2v:
            word2id[w] = len(word2id)
            vectors.append(torch.tensor(w2v[w]))

    return word2id, torch.stack(vectors)


# Dataset
class SST2Dataset(Dataset):
    def __init__(self, df, word2id, emb):
        self.samples = []
        for _, r in df.iterrows():
            ids = [word2id[w] for w in r.sentence.lower().split() if w in word2id]
            if ids:
                vec = emb[ids].mean(0)
                self.samples.append((vec, torch.tensor([float(r.label)])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# 学習
def train(model, train_loader, dev_loader, epochs=10, lr=0.01):
    # 損失関数と最適化手法の定義
    loss_fn = nn.BCELoss()
    #SGD（確率的勾配降下法）
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    for e in range(epochs):
        model.train()
        # 学習用の損失と正解数を初期化
        tr_loss = tr_ok = total = 0

        for x, y in train_loader:
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_ok += ((out > 0.5) == y).sum().item()
            total += y.size(0)

        model.eval()
        dv_loss = dv_ok = dv_total = 0
        with torch.no_grad():
            for x, y in dev_loader:
                out = model(x)
                dv_loss += loss_fn(out, y).item()
                dv_ok += ((out > 0.5) == y).sum().item()
                dv_total += y.size(0)

        print(
            f"Epoch {e+1}: "
            f"Train Loss {tr_loss/len(train_loader):.4f}, "
            f"Train Acc {100*tr_ok/total:.2f}%, "
            f"Dev Acc {100*dv_ok/dv_total:.2f}%"
        )

# main
def main():
    train_df = load_sst2("./SST-2/train.tsv")
    dev_df = load_sst2("./SST-2/dev.tsv")

    vocab = build_vocab(train_df, dev_df)
    word2id, emb = load_embeddings("./GoogleNews-vectors-negative300.bin.gz", vocab)

    train_ds = SST2Dataset(train_df, word2id, emb)
    dev_ds = SST2Dataset(dev_df, word2id, emb)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=32)

    model = MeanEmbeddingClassifier(emb.size(1))
    train(model, train_loader, dev_loader)


if __name__ == "__main__":
    main()
