import numpy as np
import csv

# シグモイド関数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ロジスティック回帰：確率を返す
def logistic_regression_predict(x, w, b):
    linear = np.dot(x, w) + b
    return sigmoid(linear)

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


# 62番で保存した重みwとバイアスbをロード
w = np.load("weight.npy")
b = np.load("bias.npy")

# 語彙
train_data = text_to_feature("./SST-2/train.tsv")
vocab = sorted({word for d in train_data for word in d["feature"].keys()})
vocab_index = {word: i for i, word in enumerate(vocab)}

# 検証データ
dev_data = text_to_feature("./SST-2/dev.tsv")

# 混同行列の初期化
TP = FP = TN = FN = 0

# 1件ずつ予測しながら混同行列を更新
for sample in dev_data:
    words = sample["feature"]
    true_label = sample["label"]

    # BoW ベクトル化
    x = np.zeros(len(vocab))
    for word, count in words.items():
        if word in vocab_index:
            x[vocab_index[word]] = count

    # 予測
    p = logistic_regression_predict(x, w, b)
    pred_label = 1 if p >= 0.5 else 0

    # 混同行列の更新
    if true_label == 1 and pred_label == 1:
        TP += 1
    elif true_label == 0 and pred_label == 0:
        TN += 1
    elif true_label == 0 and pred_label == 1:
        FP += 1
    elif true_label == 1 and pred_label == 0:
        FN += 1


# ---- 出力 ----
print("=== 混同行列（Confusion Matrix）===")
print(f"TN (正しく0): {TN}")
print(f"FP (誤って1): {FP}")
print(f"FN (誤って0): {FN}")
print(f"TP (正しく1): {TP}\n")

#　適合率 TP / (TP + FP)
precision = TP /(TP + FP) 
print(f"適合率: {precision}")

# 再現率 TP / (TP + FN)
recall = TP / (TP + FN)
print(f"再現率: {recall}")

#F-スコア 2/(1/適合率 + 1/再現率)
f_score = 2 / (1 / precision + 1 / recall)
print(f"F-スコア:{f_score}")