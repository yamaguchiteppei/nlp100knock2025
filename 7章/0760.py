import pandas as pd  

#　データの読み込み
df_train = pd.read_csv('./SST-2/train.tsv', sep ='\t')
df_dev = pd.read_csv('./SST-2/dev.tsv', sep ='\t')

#　ラベルの分布を確認(.value_counts()で各ラベルの出現回数をカウント)
print(df_train["label"].value_counts())
print(df_dev["label"].value_counts())