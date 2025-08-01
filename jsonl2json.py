# import json

# input_file = '/mnt/petrelfs/mengfanqing/MM_vocab/playground/llava_v1_5_mix665k_filtered.jsonl'
# output_file = '/mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/playground/llava_v1_5_mix665k_filtered.json'

# # Read .jsonl file and convert to standard JSON format
# with open(input_file, 'r') as infile:
#     lines = infile.readlines()

# # Parse each line as a JSON object
# data = [json.loads(line) for line in lines]

# # Write to standard JSON file
# with open(output_file, 'w') as outfile:
#     json.dump(data, outfile, indent=4)

# print(f"File converted and saved to {output_file}")
import json
import logging

# Set up logging configuration
log_file = '/mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/playground/json_diff.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# Load JSON file content
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Compare two JSON objects
def compare_json(json1, json2, path=""):
    differences = []
    
    # Check keys that exist in json1 but not in json2
    for key in json1:
        if key not in json2:
            differences.append(f"Key '{key}' is missing in second JSON (at {path + '.' + key if path else key})")
        else:
            if isinstance(json1[key], dict) and isinstance(json2[key], dict):
                differences.extend(compare_json(json1[key], json2[key], path + '.' + key if path else key))
            elif json1[key] != json2[key]:
                differences.append(f"Different values for key '{key}' (at {path + '.' + key if path else key}): {json1[key]} != {json2[key]}")
    
    # Check keys that exist in json2 but not in json1
    for key in json2:
        if key not in json1:
            differences.append(f"Key '{key}' is missing in first JSON (at {path + '.' + key if path else key})")
    
    return differences

# File paths
file1 = '/mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/output/llava_v1_5_mix665k.json'
file2 = '/mnt/petrelfs/mengfanqing/LLaVA_vocab/LLaVA-main/playground/llava_v1_5_mix665k_filtered.json'

# Load two JSON files
json1 = load_json(file1)
json2 = load_json(file2)

# Compare two JSON files
differences = compare_json(json1, json2)

# Write differences to log file
if differences:
    for diff in differences:
        logging.info(diff)  # Log each difference
else:
    logging.info("The JSON files are identical.")

print(f"Comparison complete. Differences logged to {log_file}")
