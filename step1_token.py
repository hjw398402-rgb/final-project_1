import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

dataset = load_dataset("smilegate-ai/kor_unsmile")
model_name = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, tokenizer):
        self.tokenizer = tokenizer
        self.sentences = raw_dataset["문장"]
        # clean=1이면 정상(0), clean=0이면 혐오(1)
        self.labels = [
            0 if row["clean"] == 1 else 1
            for row in raw_dataset
        ]

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

train_dataset = CustomDataset(dataset["train"], tokenizer)
valid_dataset = CustomDataset(dataset["valid"], tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  # 혐오 / 정상 딱 2개
)

training_args = TrainingArguments(
    output_dir="./results_binary",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=5,  # 3→5로 늘림
    weight_decay=0.01,
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