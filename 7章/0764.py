import numpy as np
import csv

# ---- シグモイド関数 ----
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ---- ロジスティック回帰 予測関数 ----
def logistic_regression_predict(x, w, b):
    linear = np.dot(x, w) + b
    p = sigmoid(linear)
    return p


# ---- 特徴ベクトル作成（課題61） ----
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


# 62番で保存した w, b をロード
w = np.load("weight.npy")         # 重みベクトル
b = np.load("bias.npy")       # バイアス値

print("重みベクトル", w)
print("バイアス:", b)

# 語彙（vocab）作成 
train_data = text_to_feature("./SST-2/train.tsv")
vocab = sorted({word for d in train_data for word in d["feature"].keys()})
vocab_index = {w: i for i, w in enumerate(vocab)}


# dev.tsv の読み込み
dev_data = text_to_feature("./SST-2/dev.tsv")

sample = dev_data[0]
words = sample["feature"]

# ---- BoW ベクトル化 ----
x = np.zeros(len(vocab))
for word, count in words.items():
    if word in vocab_index:
        x[vocab_index[word]] = count

# ---- 予測 ----
p = logistic_regression_predict(x, w, b)

print("\n=== 検証データ 1件目の予測 ===")
print(f"P(label=1 | x): {p} (positiveだと予測する確率)")
print(f"P(label=0 | x): {1 - p} (negativeだと予測する確率))")
