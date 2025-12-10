import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

# データ読み込み (comment=":" → : capital-common-countries のような見出し行を無視)

df = pd.read_csv(
    "questions-words.txt",
    sep=r"\s+",
    comment=":",
    header=None
)
df.columns = ["v1", "v2", "v3", "v4"] # 4列だけ抽出（アナロジータスクの単語）

#questions-words.txtのv4に書かれた単語を重複なしで取り出し、アルファベット順に並べる
countries = sorted(set(df["v4"].values))

# モデル読み込み 
model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

# 国名ベクトル作成 
vectors = [] #国名のベクトル
valid_countries = [] #モデルに存在する国名のリスト
for c in countries:
    try:
        vectors.append(model[c])
        valid_countries.append(c)
    except KeyError:
        pass

X = np.array(vectors)

# TSNE（cosine距離）
#Word2Vec の国名ベクトルをコサイン距離で比較しつつ、2次元の散布図に落とし込むための前処理
 
tsne = TSNE(
    n_components=2,#出力次元数を 2 にする
    metric="cosine", #ベクトル同士の距離を コサイン距離 で測る設定
    random_state=0,  #t-SNE はランダム性があるため、毎回結果が変わる
    init="random",  
    perplexity=30,   #t-SNE の「近傍の広がり」を表す重要パラメータ
    n_iter_without_progress=1000,
)

#t-SNE に 国名ベクトルの集合を入力し、2 次元の座標に変換する処理
embs = tsne.fit_transform(X)

# 描画
plt.figure(figsize=(14, 10))
plt.scatter(embs[:, 0], embs[:, 1], s=10)

for i, name in enumerate(valid_countries):
    plt.text(embs[i, 0], embs[i, 1], name, fontsize=6)

plt.tight_layout()
plt.savefig("0659.png")
