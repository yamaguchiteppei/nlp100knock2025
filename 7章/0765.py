import numpy as np
import csv

# ---- シグモイド関数 ----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_predict(x, w, b):
    linear = np.dot(x, w) + b
    p = sigmoid(linear)
    return p


# 特徴ベクトル作成
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

# テキスト
text = str(input("文章を入力してください:"))

# 重みとバイアスのロード
w = np.load("weight.npy")
b = np.load("bias.npy")

print("重みベクトル", w)
print("バイアス:", b)

# train.tsv から vocab を作成 
train_data = text_to_feature("./SST-2/train.tsv")

vocab = sorted({word for d in train_data for word in d["feature"].keys()})
vocab_index = {word: i for i, word in enumerate(vocab)}

print("語彙数:", len(vocab))

# BoW ベクトル化
feature = {}
for w_ in text.split():
    feature[w_] = feature.get(w_, 0) + 1

x = np.zeros(len(vocab))
for word, count in feature.items():
    if word in vocab_index:
        x[vocab_index[word]] = count

# 予測
p = logistic_regression_predict(x, w, b)
pred = 1 if p >= 0.53 else 0 #←閾値を0.5から0.53に変更

print("\n=== 与えられた文章の予測 ===")
print("文章:", text)
print(f"P(label=1|x): {p:.4f}")
print(f"P(label=0|x): {1 - p:.4f}")
print("予測ラベル:", pred)
