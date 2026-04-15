from datasets import load_dataset

dataset = load_dataset("smilegate-ai/kor_unsmile")
train = dataset['train']

clean_count = sum(1 for row in train if row['clean'] == 1)
hate_count = len(train) - clean_count

print(f"전체: {len(train)}")
print(f"clean: {clean_count}")
print(f"혐오:  {hate_count}")
print(f"비율:  혐오 {hate_count} : clean {clean_count} = {hate_count/clean_count:.2f} : 1")
