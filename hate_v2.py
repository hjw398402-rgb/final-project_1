from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "C:/workspace/finalproject/results/checkpoint-5628"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

text = "시발 년아"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
probs = torch.sigmoid(logits)
print(probs)