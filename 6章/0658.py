import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("questions-words.txt", sep=" ")
df = df.reset_index()
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)
df = df.iloc[:5030]
country = list(set(df["v4"].values))

# モデル読み込み
model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

countryVec = [] #国名のベクトル
countryName = []#国名のリスト
for c in country:
    countryVec.append(model[c])#国名のベクトルを取得
    countryName.append(c) #国名をリストに追加

X = np.array(countryVec)

#X のベクトル同士の距離をもとに階層的クラスタリングを実行。
#X:国のベクトル一覧,method:クラスタ間の距離の計算に「Ward 法」を使用,metric:距離の計算に「ユークリッド距離」を使用
linkage_result = linkage(X, method="ward", metric="euclidean")

#描画の設定を行い、図のキャンバスを作る
plt.figure(num=None, figsize=(16, 9), dpi=200, facecolor="w", edgecolor="k")

#階層クラスタリング結果をデンドログラム（樹形図）として描画
dendrogram(linkage_result, labels=countryName)

plt.savefig("dendrogram.png")