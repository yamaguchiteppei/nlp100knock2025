import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer

# シグモイド関数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ラベル予測
def logistic_regression_predict(x, w, b):
    p = sigmoid(np.dot(x, w) + b)
    return 1 if p >= 0.5 else 0

# 特徴抽出 
def text_to_feature(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for text, label in reader:
            data_list.append({"text": text, "label": int(label)})
    return data_list


# 62番で保存した w, b
w = np.load("weight.npy")
b = np.load("bias.npy")

# train 語彙を作る
train_data = text_to_feature("./SST-2/train.tsv")
count = CountVectorizer()
train_texts = [d["text"] for d in train_data]
count.fit(train_texts)

# dev データ
dev_data = text_to_feature("./SST-2/dev.tsv")
sample = dev_data[0]  # dev の 1件目
dev_vector = count.transform([sample["text"]]).toarray()[0]  # BoW に変換

# 予測 
pred_label = logistic_regression_predict(dev_vector, w, b)

print("=== 課題63：ロジスティック回帰による予測 ===")
print("文章:", sample["text"])
print("予測ラベル:", pred_label)
print("正解ラベル:", sample["label"])
