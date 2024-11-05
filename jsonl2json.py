# import json

# input_file = '/mnt/petrelfs/mengfanqing/MM_vocab/playground/llava_v1_5_mix665k_filtered.jsonl'
# output_file = '/mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/playground/llava_v1_5_mix665k_filtered.json'

# # 读取 .jsonl 文件并转换为标准 JSON 格式
# with open(input_file, 'r') as infile:
#     lines = infile.readlines()

# # 将每一行解析为一个 JSON 对象
# data = [json.loads(line) for line in lines]

# # 写入标准 JSON 文件
# with open(output_file, 'w') as outfile:
#     json.dump(data, outfile, indent=4)

# print(f"File converted and saved to {output_file}")
import json
import logging

# 设置日志配置
log_file = '/mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/playground/json_diff.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# 读取 JSON 文件内容
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 比较两个 JSON 对象
def compare_json(json1, json2, path=""):
    differences = []
    
    # 检查 json1 中有而 json2 中没有的键
    for key in json1:
        if key not in json2:
            differences.append(f"Key '{key}' is missing in second JSON (at {path + '.' + key if path else key})")
        else:
            if isinstance(json1[key], dict) and isinstance(json2[key], dict):
                differences.extend(compare_json(json1[key], json2[key], path + '.' + key if path else key))
            elif json1[key] != json2[key]:
                differences.append(f"Different values for key '{key}' (at {path + '.' + key if path else key}): {json1[key]} != {json2[key]}")
    
    # 检查 json2 中有而 json1 中没有的键
    for key in json2:
        if key not in json1:
            differences.append(f"Key '{key}' is missing in first JSON (at {path + '.' + key if path else key})")
    
    return differences

# 文件路径
file1 = '/mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/output/llava_v1_5_mix665k.json'
file2 = '/mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/playground/llava_v1_5_mix665k_filtered.json'

# 加载两个 JSON 文件
json1 = load_json(file1)
json2 = load_json(file2)

# 比较两个 JSON 文件
differences = compare_json(json1, json2)

# 将差异写入日志文件
if differences:
    for diff in differences:
        logging.info(diff)  # 记录每个差异
else:
    logging.info("The JSON files are identical.")

print(f"Comparison complete. Differences logged to {log_file}")
