import pandas as pd
import numpy as np
from gensim.models import KeyedVectors as kv
from tqdm import tqdm

# ------------------------------
# ① GoogleNews モデルをロード
# ------------------------------
print("Loading model...")
model = kv.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)

# ------------------------------
# ② 全単語のベクトルを numpy に変換（高速化の要）
# ------------------------------
print("Preparing vectors...")

# 語彙一覧
all_words = np.array(list(model.key_to_index.keys()))

# 形状 (3000000, 300)
all_vecs = model.vectors  

# 内積計算に備えて正規化
all_norm = np.linalg.norm(all_vecs, axis=1, keepdims=True)
all_vecs_unit = all_vecs / all_norm

# ------------------------------
# ③ 高速類推（analogies）関数
# ------------------------------
def fast_analogy(v1, v2, v3):
    # 単語が辞書に無ければ None を返す
    if any(w not in model for w in [v1, v2, v3]):
        return None, None

    # ベクトルを取得して正規化
    target = model[v2] + model[v3] - model[v1]
    target = target / np.linalg.norm(target)

    # コサイン類似度 = 内積
    sims = np.dot(all_vecs_unit, target)

    # 最も類似度の高い単語を取得
    best = np.argmax(sims)

    return all_words[best], sims[best]


# ------------------------------
# ④ question-words を読み込み
# ------------------------------
df = pd.read_csv("question-words.txt", sep=" ", header=None)
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)

# ------------------------------
# ⑤ DataFrame に類推を適用（高速）
# ------------------------------
sim_words = []
sim_scores = []

print("Computing analogies...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    w, s = fast_analogy(row["v1"], row["v2"], row["v3"])
    sim_words.append(w)
    sim_scores.append(s)

df["simWord"] = sim_words
df["simScore"] = sim_scores

# ------------------------------
# ⑥ 出力保存
# ------------------------------
df.to_csv("ans54.txt", sep=" ", index=False, header=None)

print("Done! Saved to ans54.txt")
