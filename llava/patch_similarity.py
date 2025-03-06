import torch
import torch.nn.functional as F
from PIL import Image
from llava.mm_utils import process_images, get_model_name_from_path
import os

def compute_least_squares_representation(
    image_features: torch.Tensor,
    vocab_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  
    vocab_embeddings = vocab_embeddings.to(dtype=torch.float32)
    image_features = image_features.to(dtype=torch.float32)

    # 计算余弦相似度
    image_features_norm = F.normalize(image_features, dim=1)
    vocab_embeddings_norm = F.normalize(vocab_embeddings, dim=1)
    similarity = torch.mm(image_features_norm, vocab_embeddings_norm.t())

    # 选择前2个最相似的词表嵌入
    top_k = 2
    top_indices = torch.topk(similarity, top_k, dim=1).indices

    # 根据最相似的词表嵌入计算最小二乘解
    selected_vocab_embeddings = vocab_embeddings[top_indices]
    coefficients = []
    mse_losses = []
    for i in range(image_features.size(0)):
        selected_embeddings = selected_vocab_embeddings[i].T
        coeff = torch.linalg.lstsq(selected_embeddings, image_features[i].unsqueeze(1)).solution.T
        coefficients.append(coeff)
        
        # 使用 coefficients 线性组合 selected_vocab_embeddings 重建 image_features
        reconstructed_feature = torch.mm(coeff, selected_embeddings.T)
        
        # 计算每个 patch 的 MSE 误差
        mse_loss = F.mse_loss(image_features[i], reconstructed_feature).item()
        mse_losses.append(mse_loss)
    
    coefficients = torch.cat(coefficients, dim=0)
    mse_losses = torch.tensor(mse_losses)

    return coefficients, mse_losses, top_indices
    

def analyze_patch_similarity(
    model,
    image_processor,
    image_path: str,
    tokenizer,
    output_file: str = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    分析图像patch的词表表示系数和相似度
    
    参数:
        model: LLaVA模型
        image_processor: 图像处理器
        image_path: 输入图像路径
        tokenizer: 分词器，用于获取词表
        output_file: 输出文件路径
    
    返回:
        每个patch的词表表示系数和重建损失
    """
    # 获取处理后的图像
    image = Image.open(image_path).convert('RGB')
    processed_tensor = process_images([image], image_processor, model.config)[0]
    
    with torch.inference_mode():
        image_features = model.get_vision_tower()(
            processed_tensor.unsqueeze(0).half().cuda()
        )
        image_features = model.get_model().mm_projector(image_features)
        image_features = image_features.squeeze(0)
    
    vocab_embeddings = model.get_model().embed_tokens.weight.to(device=image_features.device)
    
    # 计算最小二乘表示
    representation_coefficients, mse_losses, top_indices = compute_least_squares_representation(image_features, vocab_embeddings)
    
    # 计算余弦相似度
    image_features_norm = F.normalize(image_features, dim=1)
    vocab_embeddings_norm = F.normalize(vocab_embeddings, dim=1)
    similarity = torch.mm(image_features_norm, vocab_embeddings_norm.t())
    
    # 准备输出
    output_lines = []
    
    # 分析每个patch
    top_k = 10
    for patch_idx in range(representation_coefficients.shape[0]):
        output_lines.append(f"\n=== Patch {patch_idx} 分析 ===")
        
        # 使用 compute_least_squares_representation 中的 top_indices
        top_indices_patch = top_indices[patch_idx]
        output_lines.append(f"Patch {patch_idx} Top Indices: {top_indices_patch.tolist()}")
        
        # 最小二乘系数的top-k
        coeffs = representation_coefficients[patch_idx]
        # 确保 top_k 不超过 coeffs 的大小
        actual_top_k = min(top_k, coeffs.size(0))
        top_values, top_indices_patch = coeffs, top_indices_patch
        output_lines.append(f"\n最小二乘法 Top {actual_top_k} 系数:")
        for value, idx in zip(top_values, top_indices_patch):
            token = tokenizer.decode([idx])
            output_lines.append(f"Token: {token}, 系数: {value:.4f}")
        
        # 余弦相似度的top-k
        sim_values, sim_indices = torch.topk(similarity[patch_idx], top_k)
        output_lines.append(f"\n余弦相似度 Top {top_k}:")
        for value, idx in zip(sim_values, sim_indices):
            token = tokenizer.decode([idx])
            output_lines.append(f"Token: {token}, 相似度: {value:.4f}")
        
        # 输出每个 patch 的 MSE 误差
        output_lines.append(f"Patch {patch_idx} MSE Loss: {mse_losses[patch_idx]:.6f}")
    
    output_lines.append(f"\n总重建损失: {mse_losses.sum():.6f}")
    
    # 将结果写入文件或打印到控制台
    output_text = '\n'.join(output_lines)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
    else:
        print(output_text)
    
    return representation_coefficients, mse_losses, top_indices

if __name__ == "__main__":
    import argparse
    from llava.model.builder import load_pretrained_model
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="LLaVA模型路径")
    parser.add_argument("--image-path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output-file", type=str, help="输出文件路径")
    args = parser.parse_args()
    
    # 加载模型
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    print(model)
    
    # 计算相似度
    coefficients, mse_losses, top_indices = analyze_patch_similarity(
        model,
        image_processor,
        args.image_path,
        tokenizer,
        args.output_file,
    )

    