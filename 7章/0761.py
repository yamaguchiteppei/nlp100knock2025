import csv

#特徴量抽出
def text_to_feature(file_path):
    data_list = []   # データ格納用リスト

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # ヘッダー行を飛ばす

        for text, label in reader:
            words = text.split()
            feature = {}

            for word in words:
                feature[word] = feature.get(word, 0) + 1

            data_list.append({
                "text": text,
                "label": label,
                "feature": feature #feature辞書を追加
            })

    return data_list


print("train.tsv において")
train_list = text_to_feature("./SST-2/train.tsv")
print(train_list[0])  # 先頭だけ確認

print("\ndev.tsv において")
dev_list = text_to_feature("./SST-2/dev.tsv")
print(dev_list[0]) #先頭だけ確認
