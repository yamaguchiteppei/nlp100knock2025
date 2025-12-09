import pandas as pd
from gensim.models import KeyedVectors as kv
from tqdm import tqdm


def culcCosSim(row):
    global model
    return model.similarity(row["Word 1"], row["Word 2"])


tqdm.pandas()

model = kv.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)

df = pd.read_csv("wordsim353/combined.csv")

df["cosSim"] = df.progress_apply(culcCosSim, axis=1)

print(df[["Human (mean)", "cosSim"]].corr(method="spearman"))
# 出力
#              Human (mean)    cosSim
#Human (mean)      1.000000  0.700017
#cosSim            0.700017  1.000000