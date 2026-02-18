#①ライブラリ
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

#②デバイス
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

#③モデル
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
model_ref = AutoModelForCausalLM.from_pretrained(model_name)

#④SST-2 読み込み
train_df = pd.read_csv("./train.tsv", sep="\t")

#⑤ラベル変換
def label_to_text(label):
    return "positive" if label == 1 else "negative"

#⑥DPO用データ作成
def build_dpo_dataset(df):

    prompts = []
    chosen = []
    rejected = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = row["label"]

        prompt = f"Review: {text}\nSentiment:"

        correct = " " + label_to_text(label)
        wrong = " negative" if label == 1 else " positive"

        prompts.append(prompt)
        chosen.append(correct)
        rejected.append(wrong)

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosen,
        "rejected": rejected
    })

#⑦ Dataset生成
dpo_dataset = build_dpo_dataset(train_df)

#⑧ DPO設定
training_args = DPOConfig(
    output_dir="./dpo_model",
    per_device_train_batch_size=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=50
)

#⑨ Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=model_ref,
    args=training_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer
)

#⑩ 学習
trainer.train()

#⑪ 推論関数
model.to(device)
model.eval()

def predict(text):

    prompt = f"Review: {text}\nSentiment:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
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

#⑫ 評価
dev_df = pd.read_csv("./dev.tsv", sep="\t")

correct = 0
total = 0

for _, row in dev_df.iterrows():
    pred = predict(row["sentence"])
    if pred is not None:
        total += 1
        if pred == row["label"]:
            correct += 1

print("Dev Accuracy:", correct/total)
