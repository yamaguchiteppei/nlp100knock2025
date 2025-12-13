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


# ロジスティック回帰モデル
#シグモイド関数　1/(1+exp(-z))　
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ロジスティック回帰の学習 (X: 特徴行列, y: ラベルベクトル, lr: 学習率, epochs: エポック数)
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


#  学習
w, b = logistic_regression_train(X, y)



# 学習結果の確認

print("学習完了!")
print("重みベクトル (w) ", w)
np.save("weight.npy", w)
print("バイアス (b):", b)
np.save("bias.npy", b)
