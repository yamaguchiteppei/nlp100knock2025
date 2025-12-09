#単語埋め込み用のライブラリであるgensimを持ってくる
from gensim.models import KeyedVectors as kv

#word2vecのモデルを読み込む
model = kv.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)

#model.similarity(A,B)でAとBの類似度を計算することができる。
print(model.similarity("United_States", "U.S."))

"""実行結果
0.73107743
"""