import torch
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
from torch.nn.functional import cosine_similarity

# モデルとトークナイザ
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()  # 推論モード

sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.", #crap:くだらない
    "The movie was full of rubbish."　#rubbish:くだらない,ゴミ
]

cls_vectors = []

# 各文の [CLS] ベクトルを取得
with torch.no_grad():
    for s in sentences:
        inputs = tokenizer(s, return_tensors="pt")
        outputs = model(**inputs)
        # 最終層の [CLS] トークン（batch=0, token=0）
        cls_vec = outputs.last_hidden_state[0, 0]
        cls_vectors.append(cls_vec)

# 全組み合わせでコサイン類似度を計算
for (i, j) in combinations(range(len(sentences)), 2):
    sim = cosine_similarity(
        cls_vectors[i].unsqueeze(0),
        cls_vectors[j].unsqueeze(0)
    ).item()
    print(f"({i}, {j}) {sim:.4f}")
