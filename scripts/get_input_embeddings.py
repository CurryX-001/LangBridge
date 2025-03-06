import torch
from transformers import (AutoModel, GenerationConfig, AutoTokenizer,LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
import json

# Part 1: 获取模型嵌入矩阵
def get_model_embeddings(model_name: str, save_path: str) -> None:
    """
    获取预训练模型的嵌入矩阵并保存
    """
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    embedding_matrix = model.get_input_embeddings().weight
    torch.save(embedding_matrix, save_path)
    print(f"嵌入矩阵已保存至: {save_path}")

# Part 2: 处理vocab和嵌入矩阵
def process_embeddings(embedding_path: str, vocab_path: str, save_path: str) -> None:
    """
    根据vocab筛选并保存对应的嵌入向量
    """
    # 加载嵌入矩阵
    embedding_matrix = torch.load(embedding_path)
    if hasattr(embedding_matrix, 'weight'):
        embedding_matrix = embedding_matrix.weight
    
    # 加载vocab数据
    with open(vocab_path, 'r') as file:
        vocab_data = json.load(file)
    
    # 获取并验证索引
    indices = list(vocab_data.values())
    max_index = embedding_matrix.size(0)
    if max(indices) >= max_index:
        raise ValueError(f"vocab索引超出范围，最大允许索引为{max_index-1}")
    
    # 选择对应的嵌入向量
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    mm_vocab = torch.index_select(embedding_matrix, 0, indices_tensor)
    
    # 保存结果
    torch.save(mm_vocab, save_path)
    print(f"处理后的vocab嵌入向量已保存至: {save_path}")

if __name__ == "__main__":
    # 示例使用
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    embedding_save_path = "/mnt/data/jiaqi.liao/Codebook/Qwen2.5-14B-Instruct_embedding_matrix.pt"
    
    # 获取嵌入矩阵
    get_model_embeddings(model_name, embedding_save_path)
    
    # 处理vocab和嵌入
    vocab_path = "/mnt/data/jiaqi.liao/Codebook/19200_Qwen_sub_llava_share_intersect_llama_qwen.json"
    final_save_path = "/mnt/data/jiaqi.liao/Codebook/Qwen2.5-14B-Instruct_embedding_matrix_19200.pt"
    process_embeddings(embedding_save_path, vocab_path, final_save_path)