import torch
import torch.nn as nn
import pandas as pd
from typing import Dict, List, Set
from gensim.models import KeyedVectors


# データ読み込み
def load_sst2(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")

# 語彙構築
def build_vocabulary(dfs: List[pd.DataFrame]) -> Set[str]:
    vocab = set()
    for df in dfs:
        for s in df["sentence"]:
            vocab.update(s.lower().split())
    return vocab


# 単語埋め込み
def load_embeddings(
    model_path: str, vocab: Set[str]
) -> tuple[Dict[str, int], torch.Tensor]:
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    word_to_id = {"<PAD>": 0}
    vectors = [torch.zeros(model.vector_size)]

    for w in vocab:
        if w in model.key_to_index:
            word_to_id[w] = len(word_to_id)
            vectors.append(torch.tensor(model[w]))

    return word_to_id, torch.stack(vectors)

# 前処理
def text_to_ids(text: str, word_to_id: Dict[str, int]) -> List[int]:
    return [word_to_id[w] for w in text.lower().split() if w in word_to_id]

# データセット構築
def build_dataset(df: pd.DataFrame, word_to_id: Dict[str, int]) -> List[Dict]:
    data = []
    for _, row in df.iterrows():
        ids = text_to_ids(row["sentence"], word_to_id)
        if ids:
            data.append(
                {
                    "input_ids": torch.tensor(ids),
                    "label": torch.tensor([float(row["label"])]),
                }
            )
    return data
# 平均埋め込み計算
def mean_embeddings(
    data: List[Dict], emb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = [], []
    for d in data:
        X.append(emb[d["input_ids"]].mean(dim=0))
        y.append(d["label"])
    return torch.stack(X), torch.cat(y)

# モデル
class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# main
def main():
    train_df = load_sst2("./SST-2/train.tsv")
    dev_df = load_sst2("./SST-2/dev.tsv")

    vocab = build_vocabulary([train_df, dev_df])
    word_to_id, emb = load_embeddings(
        "./GoogleNews-vectors-negative300.bin.gz", vocab
    )

    train = build_dataset(train_df, word_to_id)
    dev = build_dataset(dev_df, word_to_id)

    X_train, y_train = mean_embeddings(train, emb)
    X_dev, y_dev = mean_embeddings(dev, emb)

    model = MeanEmbeddingClassifier(emb.size(1))

    print(model)
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)


if __name__ == "__main__":
    main()
