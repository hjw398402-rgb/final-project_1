with open('./Curse-detection-data-master/dataset.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

clean = sum(1 for line in lines if line.strip().split('|')[-1] == '0')
hate = sum(1 for line in lines if line.strip().split('|')[-1] == '1')

print(f"전체: {len(lines)}")
print(f"clean: {clean}")
print(f"혐오:  {hate}")



