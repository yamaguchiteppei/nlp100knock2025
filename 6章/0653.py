#単語埋め込み用のgensimライブラリを持ってくる。
from gensim.models import KeyedVectors as kv

#word2vecをロードする。
model = kv.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True
)
#most.similar(positive=[A,B],negative=[C])でA+B-Cのベクトルの数値に最も近い単語を表示する。
result = model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10)
print(result)

"""実行結果
[('Greece', 0.6898480653762817), ('Aristeidis_Grigoriadis', 0.560684859752655), ('Ioannis_Drymonakos', 0.5552908778190613), ('Greeks', 0.545068621635437), ('Ioannis_Christou', 0.5400862097740173), ('Hrysopiyi_Devetzi', 0.5248444676399231), ('Heraklio', 0.5207759737968445), ('Athens_Greece', 0.516880989074707), ('Lithuania', 0.5166865587234497), ('Iraklion', 0.5146791338920593)]
"""