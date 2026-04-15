from datasets import load_dataset

# 1. 여기서 'dataset'이라는 변수에 값을 담아줍니다. (이 줄이 꼭 먼저 실행되어야 함!)
dataset = load_dataset("smilegate-ai/kor_unsmile")

# 2. 위에서 잘 담겼으니 이제 에러 없이 출력될 거예요.
print("--- 데이터셋 구조 확인 ---")
print(dataset)

# 3. 훈련 데이터의 첫 번째 문장 확인
print("\n--- 첫 번째 데이터 샘플 ---")
print(dataset['train'][0])

