import numpy as np
from gensim.models import KeyedVectors


# 設定
W2V_PATH = "./GoogleNews-vectors-negative300.bin.gz"
EMBEDDING_DIM = 300


# word2vec の読み込み
print("Loading word2vec model...")
w2v = KeyedVectors.load_word2vec_format(
    W2V_PATH,
    binary=True
)

vocab_size = len(w2v.key_to_index)
print(f"Vocabulary size: {vocab_size}")

# 単語埋め込み行列の作成
embedding_matrix = np.zeros((vocab_size + 1, EMBEDDING_DIM), dtype=np.float32)

# トークンとIDの対応表
token_to_id = {"<PAD>": 0}
id_to_token = {0: "<PAD>"}

# 埋め込みを2行目以降に格納
for idx, word in enumerate(w2v.key_to_index.keys(), start=1):
    embedding_matrix[idx] = w2v[word]
    token_to_id[word] = idx
    id_to_token[idx] = word

# 確認
print("Embedding matrix shape:", embedding_matrix.shape)
print("PAD vector:", embedding_matrix[0][:5])  # 先頭5次元だけ表示
print("Example token:", id_to_token[1])
