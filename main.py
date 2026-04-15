from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

# React 연결 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
MODEL_PATH = "Jinwoo1251a/best_model_v4"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

class TextInput(BaseModel):
    text: str

@app.post("/check")
def check_hate(input: TextInput):
    inputs = tokenizer(
        input.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = F.softmax(logits, dim=-1).squeeze(0)
    hate_score = probs[1].item()
    return {"is_hate": hate_score >= 0.68, "score": hate_score}