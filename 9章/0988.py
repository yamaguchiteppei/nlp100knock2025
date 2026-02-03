import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# モデル読み込み
# =========================
model_dir = "finetuned-sst2"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


# =========================
# 予測したい文
# =========================
sentences = [
    "The movie was full of incomprehensibilities.",
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish."
]


# =========================
# 推論
# =========================
with torch.no_grad():
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    outputs = model(**enc)
    preds = torch.argmax(outputs.logits, dim=1)


# =========================
# 表示
# =========================
label_map = {0: "NEGATIVE", 1: "POSITIVE"}

for s, p in zip(sentences, preds):
    print(f"{label_map[p.item()]}\t{s}")
