import torch
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
from torch.nn.functional import cosine_similarity

# 文
sentences = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]

# BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# 文ベクトルを保存
sentence_vectors = []

with torch.no_grad():
    for s in sentences:
        inputs = tokenizer(s, return_tensors="pt")
        outputs = model(**inputs)

        # 最終層の埋め込み [1, seq_len, hidden]
        last_hidden = outputs.last_hidden_state

        # mean pooling（トークン方向に平均）
        sent_vec = last_hidden.mean(dim=1).squeeze(0)  # [768]

        sentence_vectors.append(sent_vec)

# コサイン類似度
for (i, j) in combinations(range(len(sentences)), 2):
    sim = cosine_similarity(
        sentence_vectors[i].unsqueeze(0),
        sentence_vectors[j].unsqueeze(0)
    ).item()
    print(f"({i}, {j}) {sim:.4f}")
