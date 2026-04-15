import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# ─────────────────────────────────────────
# 1. kor_unsmile
# ─────────────────────────────────────────
dataset = load_dataset("smilegate-ai/kor_unsmile")

train_data = []
valid_data = []

for row in dataset["train"]:
    train_data.append({"문장": row["문장"], "label": 0 if row["clean"] == 1 else 1})
for row in dataset["valid"]:
    valid_data.append({"문장": row["문장"], "label": 0 if row["clean"] == 1 else 1})

print(f"kor_unsmile train: {len(train_data)}, val: {len(valid_data)}")

# ─────────────────────────────────────────
# 2. HateScore
# ─────────────────────────────────────────
df_hate = pd.read_csv('./hatescore-korean-hate-speech-main/HateScore.csv')

# 90% train, 10% val로 분리
df_hate_train = df_hate.sample(frac=0.9, random_state=42)
df_hate_val   = df_hate.drop(df_hate_train.index)

for _, row in df_hate_train.iterrows():
    label = 0 if row['macrolabel'] == 'Clean' else 1
    train_data.append({"문장": str(row['comment']), "label": label})

for _, row in df_hate_val.iterrows():
    label = 0 if row['macrolabel'] == 'Clean' else 1
    valid_data.append({"문장": str(row['comment']), "label": label})

print(f"HateScore 추가 후 train: {len(train_data)}, val: {len(valid_data)}")

# ─────────────────────────────────────────
# 3. Curse-detection
# ─────────────────────────────────────────
with open('./Curse-detection-data-master/dataset.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 90% train, 10% val로 분리
total = len(lines)
val_size = int(total * 0.1)
curse_val_lines   = lines[:val_size]
curse_train_lines = lines[val_size:]

for line in curse_train_lines:
    parts = line.strip().split('|')
    if len(parts) == 2 and parts[1].strip() in ['0', '1']:
        train_data.append({"문장": parts[0].strip(), "label": int(parts[1].strip())})

for line in curse_val_lines:
    parts = line.strip().split('|')
    if len(parts) == 2 and parts[1].strip() in ['0', '1']:
        valid_data.append({"문장": parts[0].strip(), "label": int(parts[1].strip())})

print(f"Curse-detection 추가 후 train: {len(train_data)}, val: {len(valid_data)}")

# 최종 비율 확인
train_clean = sum(1 for d in train_data if d['label'] == 0)
train_hate  = sum(1 for d in train_data if d['label'] == 1)
val_clean   = sum(1 for d in valid_data if d['label'] == 0)
val_hate    = sum(1 for d in valid_data if d['label'] == 1)

print(f"\n최종 train  → clean: {train_clean}, 혐오: {train_hate}")
print(f"최종 val    → clean: {val_clean},   혐오: {val_hate}")

# ─────────────────────────────────────────
# 4. 토크나이저 & 데이터셋
# ─────────────────────────────────────────
model_name = "beomi/KcELECTRA-base-v2022"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.sentences = [d["문장"] for d in data]
        self.labels    = [d["label"] for d in data]

    def __getitem__(self, idx):
        inputs = self.tokenizer(
            self.sentences[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.sentences)

train_dataset = CustomDataset(train_data, tokenizer)
valid_dataset = CustomDataset(valid_data, tokenizer)

# ─────────────────────────────────────────
# 5. 모델 & 학습
# ─────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results_electra_v3",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    logging_steps=50,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

trainer.train()
