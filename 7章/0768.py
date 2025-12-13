import numpy as np
import csv

# （特徴ベクトル作成）
def text_to_feature(file_path):
    data_list = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)

        for text, label in reader:
            words = text.split()
            feature = {}

            for w in words:
                feature[w] = feature.get(w, 0) + 1

            data_list.append({
                "text": text,
                "label": int(label),
                "feature": feature
            })

    return data_list


# シグモイド関数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ロジスティック回帰の学習
def logistic_regression_train(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        linear = np.dot(X, w) + b
        p = sigmoid(linear)

        dw = (1/n_samples) * np.dot(X.T, (p - y))
        db = (1/n_samples) * np.sum(p - y)

        w -= lr * dw
        b -= lr * db

    return w, b


# ====== ここまでが 62番（学習） ======

# 特徴ベクトル作成（課題61）
train_data = text_to_feature("./SST-2/train.tsv")

# 語彙（vocab）作成
vocab = sorted({word for d in train_data for word in d["feature"].keys()})
vocab_index = {word: i for i, word in enumerate(vocab)}

# BoW を数値行列に変換
X = np.zeros((len(train_data), len(vocab)))

for i, data in enumerate(train_data):
    for word, count in data["feature"].items():
        X[i, vocab_index[word]] = count

y = np.array([d["label"] for d in train_data])


# 62番で保存した重みwとバイアスbをロード
w = np.load("weight.npy")
b = np.load("bias.npy")

# ここからが 65番：重みトップ20 / ワースト20 ======

# w を小さい順に並べたインデックス
indices_sorted = np.argsort(w)   # 小さい → 大きい

# 重みが大きい（ポジティブ寄り）トップ20
top20_pos_idx = indices_sorted[-20:][::-1]  # 末尾20個を逆順
top20_pos = [(vocab[i], w[i]) for i in top20_pos_idx]

# 重みが小さい（ネガティブ寄り）トップ20
top20_neg_idx = indices_sorted[:20]         # 先頭20個
top20_neg = [(vocab[i], w[i]) for i in top20_neg_idx]

print("\n===== 重みが大きい特徴量トップ20（ポジ寄り）=====")
for word, weight in top20_pos:
    print(f"{word:20s} : {weight:.4f}")

print("\n===== 重みが小さい特徴量トップ20（ネガ寄り）=====")
for word, weight in top20_neg:
    print(f"{word:20s} : {weight:.4f}")
