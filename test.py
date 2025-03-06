import pandas as pd

# 读取Excel文件
df = pd.read_excel('/scratch/Codebook/llama3_8b_vocab-19200-finetune.xlsx')

# 计算prediction和answer相同的数量
correct = (df['prediction'] == df['answer']).sum()

# 计算总样本数
total = len(df)

# 计算准确率
accuracy = correct / total

print(f'准确率: {accuracy:.4f}')
print(f'正确数量: {correct}')
print(f'总样本数: {total}')