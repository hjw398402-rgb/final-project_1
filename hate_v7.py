from transformers import AutoTokenizer

# KcELECTRA 원본에서 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

# best_model_v4 폴더에 저장
tokenizer.save_pretrained("C:/workspace/finalproject/best_model_v4")