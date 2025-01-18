# import torch
# from transformers import (AutoModel, GenerationConfig, AutoTokenizer,LlamaForCausalLM,
#                           LlamaTokenizer, Qwen2ForCausalLM)
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.modeling_utils import PreTrainedModel
# from transformers.utils import ModelOutput, logging
# model = AutoModel.from_pretrained(
#     "/mnt/hwfile/gveval/mfq/Meta-Llama-3-8B-Instruct",
#     trust_remote_code=True
# )
# tokenizer = AutoTokenizer.from_pretrained( "/mnt/hwfile/gveval/mfq/Meta-Llama-3-8B-Instruct",trust_remote_code=True)
# embedding_matrix = model.get_input_embeddings().weight 

# # 将嵌入矩阵保存到文件，保存为 PyTorch 格式
# torch.save(embedding_matrix, "/mnt/petrelfs/mengfanqing/embedding_matrix_llm/Meta-Llama-3-8B-Instruct_embedding_matrix.pt")

import torch
import json

# 定义路径
embedding_matrix_path = "/mnt/petrelfs/mengfanqing/Codebook/embedding_matrix_llm/embedding_matrix_llm/Qwen2-7B-Instruct_embedding_matrix.pt"
vocab_json_path = "/mnt/petrelfs/mengfanqing/Codebook/embedding_matrix_llm/12800_Qwen_sub_llava_share_intersect_llama_qwen.json"
save_path = "/mnt/petrelfs/mengfanqing/Codebook/embedding_matrix_llm/Qwen2-7B-Instruct_embedding_matrix.pt_12800.pt"

# 加载嵌入矩阵（需要检查是否是状态字典或模型，按需加载）
embedding_matrix = torch.load(embedding_matrix_path)

# 如果加载的是模型或状态字典，则直接访问权重
if hasattr(embedding_matrix, 'weight'):
    embedding_matrix = embedding_matrix.weight

# 加载vocab数据
with open(vocab_json_path, 'r') as file:
    vocab_data = json.load(file)

# 获取vocab数据中的索引
indices = list(vocab_data.values())

# 确保索引有效且不超出嵌入矩阵的范围
max_index = embedding_matrix.size(0)
if max(indices) >= max_index:
    raise ValueError("某些vocab索引超出了嵌入矩阵的维度范围。")

# 将索引转换为tensor，并确保是正确的数据类型
indices_tensor = torch.tensor(indices, dtype=torch.long)

# 从嵌入矩阵中选择相应的vocab嵌入
mm_vocab = torch.index_select(embedding_matrix, 0, indices_tensor)

# 保存筛选后的mm_vocab到指定路径
torch.save(mm_vocab, "/mnt/petrelfs/mengfanqing/Codebook/embedding_matrix_llm/Qwen2-7B-Instruct_embedding_matrix.pt_12800.pt")
# /mnt/petrelfs/mengfanqing/embedding_matrix_vocab_12800/Meta-Llama-3-8B-Instruct_embedding_matrix_vocab_12800.pt
print(f"筛选后的mm_vocab已保存到 {save_path}")



# from transformers import AutoTokenizer

# # 加载LLaMA3的tokenizer
# model_name = '/mnt/petrelfs/share_data/wangwenhai/llm/Meta-Llama-3-8B-Instruct'
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 查看eos_token
# print("EOS Token:", tokenizer.eos_token)
# print("EOS Token ID:", tokenizer.eos_token_id)
# print("PAD Token:", tokenizer.pad_token)
