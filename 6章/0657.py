import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

df = pd.read_csv("questions-words.txt", sep=" ")
df = df.reset_index()
df.columns = ["v1", "v2", "v3", "v4"]
df.dropna(inplace=True)
df = df.iloc[:5030]
country = list(set(df["v4"].values))

model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

countryVec = []
for c in country:
    countryVec.append(model[c])

X = np.array(countryVec)
km = KMeans(n_clusters=5, random_state=0)
y_km = km.fit_predict(X)
print(y_km)

# 出力
"""[2 0 2 3 3 4 2 3 0 4 0 2 2 2 4 3 3 2 4 1 1 1 2 3 0 0 2 1 1 4 0 0 3 3 0 3 2
 1 2 2 2 2 3 2 3 2 2 2 2 0 0 2 2 0 3 1 3 2 0 4 1 0 4 0 1 0 4 0 2 2 2 3 2 2
 3 2 4 0 2 3 0 1 3 4 2 3 2 1 2 2 0 1 3 2 2 0 3 0 1 2 0 2 0 2 3 2 1 0 2 0 3
 3 2 2 2 0]
 """