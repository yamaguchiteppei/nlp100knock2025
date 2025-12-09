import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

# ======= データ読み込み =======
df = pd.read_csv(
    "questions-words.txt",
    sep=r"\s+",
    comment=":",
    header=None
)
df.columns = ["v1", "v2", "v3", "v4"]

countries = sorted(set(df["v4"].values))

# ======= モデル読み込み =======
model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

# ======= 国名ベクトル作成 =======
vectors = []
valid_countries = []
for c in countries:
    try:
        vectors.append(model[c])
        valid_countries.append(c)
    except KeyError:
        pass

X = np.array(vectors)

# ======= TSNE（cosine距離） =======
tsne = TSNE(
    n_components=2,
    metric="cosine",
    random_state=0,
    init="random",
    perplexity=30,
    n_iter_without_progress=1000,
)

embs = tsne.fit_transform(X)

# ======= 描画 =======
plt.figure(figsize=(14, 10))
plt.scatter(embs[:, 0], embs[:, 1], s=10)

for i, name in enumerate(valid_countries):
    plt.text(embs[i, 0], embs[i, 1], name, fontsize=6)

plt.tight_layout()
plt.savefig("0659.png")
