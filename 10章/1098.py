#①ライブラリ
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

#②デバイス
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

#③モデル
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

#④SST-2 読み込み
train_df = pd.read_csv("./train.tsv", sep="\t")
dev_df   = pd.read_csv("./dev.tsv", sep="\t")

train_texts = train_df["sentence"].tolist()
train_labels = train_df["label"].tolist()

dev_texts = dev_df["sentence"].tolist()
dev_labels = dev_df["label"].tolist()

#⑤ラベルを文字列化
def label_to_text(label):
    return "positive" if label == 1 else "negative"

#⑥Dataset作成(Prompt+正解)
class SST2PromptDataset(Dataset):
    def __init__(self, texts, labels):
        self.samples = []

        for text, label in zip(texts, labels):
            prompt = f"Review: {text}\nSentiment:"
            answer = " " + label_to_text(label)

            full = prompt + answer

            self.samples.append(full)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]

        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return enc["input_ids"].squeeze(), enc["attention_mask"].squeeze()

#⑦DataLoader
train_dataset = SST2PromptDataset(train_texts, train_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

#⑧学習設定
optimizer = AdamW(model.parameters(), lr=5e-5)

#⑨ファインチューニング
model.train()

for epoch in range(3):
    total_loss = 0

    for input_ids, attention_mask in tqdm(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss:", total_loss/len(train_loader))

#⑩推論関数(学習後)
def predict(text):
    prompt = f"Review: {text}\nSentiment:"

    enc = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **enc,
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(output[0])
    answer = generated[len(prompt):].strip().lower()

    if "positive" in answer:
        return 1
    elif "negative" in answer:
        return 0
    else:
        return None

#⑪評価
correct = 0
total = 0

for text, label in zip(dev_texts, dev_labels):
    pred = predict(text)
    if pred is not None:
        total += 1
        if pred == label:
            correct += 1

print("Dev Accuracy:", correct/total)
