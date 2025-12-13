import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#  データ読み込み 
def load_tsv(file_path):
    texts, labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for text, label in reader:
            texts.append(text)
            labels.append(int(label))
    return texts, labels


# train/dev を読み込む
train_texts, train_labels = load_tsv("./SST-2/train.tsv")
dev_texts, dev_labels = load_tsv("./SST-2/dev.tsv")

# BoW ベクトル化
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)

# 試す正則化パラメータ 
C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

accuracies = []

# それぞれの C でモデル学習 → dev 正解率 
for C in C_list:
    model = LogisticRegression(max_iter=300, C=C)
    model.fit(X_train, train_labels)
    pred = model.predict(X_dev)
    acc = accuracy_score(dev_labels, pred)
    accuracies.append(acc)
    print(f"C={C} → dev accuracy={acc:.4f}")

# グラフ描画 
plt.plot(C_list, accuracies, marker="o")
plt.xscale("log")  # ログスケール推奨
plt.xlabel("正則化パラメータ C (大=正則化弱)")
plt.ylabel("正解率")
plt.title("正則化パラメータと正解率")
plt.grid(True)

plt.savefig("regularization_vs_accuracy.png")
plt.show()
