import pandas as pd
from gensim.models import KeyedVectors
from tqdm import tqdm

# アナロジーデータの読み込み 
# "capital-common-countries" セクションのみ抽出
data = []
flag = False

with open("questions-words.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith(":"):
            flag = (line == ": capital-common-countries")
            continue
        
        if flag:
            cols = line.split()
            if len(cols) == 4:
                data.append(cols)

# DataFrame化
df = pd.DataFrame(data, columns=["v1", "v2", "v3", "v4"])

# GoogleNews ベクトルモデル読み込み 
print("Loading model...")
model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# 類似度計算関数 
def culcSim(row):
    try:
        # vec(v2) - vec(v1) + vec(v3)
        #[("queen", 0.728)]
        
        word, score = model.most_similar(
            positive=[row["v2"], row["v3"]],
            negative=[row["v1"]],
            topn=1
        )[0]
        return pd.Series([word, score])
    except KeyError:
        # モデルに無い単語がある場合は NONE と -1 を返す
        return pd.Series(["NONE", -1])

tqdm.pandas()
df[["simWord", "simScore"]] = df.progress_apply(culcSim, axis=1)

# 出力 
df.to_csv("ans54.txt", sep=" ", index=False, header=False)
print("Done → ans54.txt に書き出しました")
