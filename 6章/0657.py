import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

# Word2Vec モデル読み込み
print("Loading Word2Vec model...")
model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

#  questions-words.txt から国名抽出
#  対象：: capital-common-countries セクション
countries = set()
flag = False  # 今 capital-common-countries セクションの中かどうかのフラグ

with open("questions-words.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # セクション行（: で始まる）
        if line.startswith(":"):
            # セクション切り替え
            flag = (line == ": capital-common-countries")
            continue

        # 国名抽出対象セクション以外は無視
        if not flag:
            continue

        # データ行 (例: Athens Greece Baghdad Iraq)
        parts = line.split()
        if len(parts) != 4:
            continue

        _, country1, _, country2 = parts

        countries.add(country1)
        countries.add(country2)

print(f"抽出した国名の数: {len(countries)}")


# Word2Vec に存在する国だけベクトル化
valid_countries = []
vecs = []

print("Collecting vectors...")

for c in countries:
    if c in model:  # Word2Vec に存在する単語だけ使用
        valid_countries.append(c)
        vecs.append(model[c])

X = np.vstack(vecs)
print(f"ベクトル化できた国の数: {len(valid_countries)}")


# k-means クラスタリング
k = 5
print(f"Clustering into {k} groups ...")

#scikit-learn の k-means クラスタリングのモデルを作成しているコード
km = KMeans(n_clusters=k, random_state=0)
labels = km.fit_predict(X)


#  結果を整理して表示
df_result = pd.DataFrame({
    "country": valid_countries,
    "cluster": labels
})

df_result = df_result.sort_values("cluster")

# クラスタごとに表示
for cl in range(k):
    print(f"\n=== Cluster {cl} ===")
    names = df_result[df_result["cluster"] == cl]["country"].tolist()
    print(", ".join(names))


# ==================================
# ⑥ 結果をCSV保存（必要な場合）
# ==================================
df_result.to_csv("country_cluster_result.csv", index=False)
print("\nSaved: country_cluster_result.csv")

# 実行結果
"""
ベクトル化できた国の数: 23
Clustering into 5 groups ..

=== Cluster 0 ===
Thailand, Vietnam

=== Cluster 1 ===
Australia, Japan, China

=== Cluster 2 ===
Cuba, Egypt, Iraq, Iran

=== Cluster 3 ===
Pakistan, Afghanistan

=== Cluster 4 ===
Greece, France, Spain, Finland, Russia, Germany, Switzerland, Canada, Norway, Italy, Sweden, England
"""