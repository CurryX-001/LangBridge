import torch
import torch.nn.functional as F
from PIL import Image
from llava.mm_utils import process_images, get_model_name_from_path
import os
import numpy as np

def visualize_patch_tokens(
    image: Image.Image,
    image_features: torch.Tensor,
    vocab_embeddings: torch.Tensor,
    tokenizer,
    output_path: str,
    top_k: int = 1,
    font_size: int = 10,
    text_color: str = "white",
    outline_color: str = "black",
    vis_ratio: float = 1.0,
    title: str = None,
    use_similarity: bool = False,
) -> Image.Image:
    """
    将每个patch对应的最相似token直接可视化在原始图像上
    
    参数:
        image: 原始PIL图像
        image_features: 视觉特征 [num_patches, hidden_dim]
        vocab_embeddings: 词表嵌入 [vocab_size, hidden_dim]
        tokenizer: 分词器
        output_path: 输出图像路径
        top_k: 每个patch显示的token数量
        font_size: 文字大小
        text_color: 文字颜色
        outline_color: 文字轮廓颜色
        vis_ratio: 可视化图像的缩放比例
        title: 图像标题
        use_similarity: 是否使用相似度矩阵
    
    返回:
        标注了token的图像
    """
    # 调整图像大小，保持宽高比
    new_width = int(image.width * vis_ratio)
    new_height = int(image.height * vis_ratio)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 确保数据类型匹配
    vocab_embeddings = vocab_embeddings.to(dtype=image_features.dtype)
    
    # 计算patch大小和网格尺寸
    num_patches = image_features.shape[0]
    grid_size = int(np.sqrt(num_patches))
    patch_width = new_width // grid_size
    patch_height = new_height // grid_size
    
    # 根据输入特征的维度决定是否需要计算相似度
    if image_features.shape[1] != len(tokenizer):  # 如果不是词表大小，需要计算相似度
        # 计算相似度并获取top-k tokens
        image_features_norm = F.normalize(image_features, dim=1)
        vocab_embeddings_norm = F.normalize(vocab_embeddings, dim=1)
        similarity = torch.mm(image_features_norm, vocab_embeddings_norm.t())
        top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=1)
    else:  # 如果已经是词表大小，直接使用
        top_k_values, top_k_indices = torch.topk(image_features, k=top_k, dim=1)
    
    # 创建可绘制的图像副本
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        # 对于Linux系统
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            # 对于Windows系统
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # 在每个patch上绘制对应的tokens
    for i in range(num_patches):
        row = i // grid_size
        col = i % grid_size
        x = col * patch_width
        y = row * patch_height
        
        # 获取当前patch的top-k tokens
        tokens = []
        for idx in top_k_indices[i]:
            token = tokenizer.decode(idx.item()).strip()
            tokens.append(token)
        
        # 将tokens合并成一个字符串
        text = "\n".join(tokens)
        
        # 计算文本边界框
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算文本居中位置
        text_x = x + (patch_width - text_width) // 2
        text_y = y + (patch_height - text_height) // 2
        
        # 绘制文本轮廓
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((text_x+dx, text_y+dy), text, font=font, fill=outline_color)
        
        # 绘制文本
        draw.text((text_x, text_y), text, font=font, fill=text_color)
        
        # 可选：绘制patch边界
        draw.rectangle([x, y, x+patch_width, y+patch_height], outline="red", width=1)
    
    # 保存结果
    draw_image.save(output_path)
    return draw_image

def compute_least_squares_representation(
    image_features: torch.Tensor,
    vocab_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    使用最小二乘法用词表嵌入线性表示视觉特征
    
    参数:
        image_features: 视觉特征 [num_patches, hidden_dim]
        vocab_embeddings: 词表嵌入 [vocab_size, hidden_dim]
    
    返回:
        系数矩阵 [num_patches, vocab_size]
        重建损失值
    """
    # 确保数据类型匹配并转换为float32
    vocab_embeddings = vocab_embeddings.to(dtype=torch.float32)
    image_features = image_features.to(dtype=torch.float32)
    
    # 修正最小二乘解的计算：X = I(V^TV)^(-1)V^T
    VT_V = torch.mm(vocab_embeddings.t(), vocab_embeddings)  # [hidden_dim, hidden_dim]
    VT_V_inv = torch.linalg.pinv(VT_V)  # [hidden_dim, hidden_dim]
    
    # 先计算 (V^TV)^(-1)V^T
    right_term = torch.mm(VT_V_inv, vocab_embeddings.t())  # [hidden_dim, vocab_size]
    
    # 然后左乘图像特征：I(V^TV)^(-1)V^T
    coefficients = torch.mm(image_features, right_term)  # [num_patches, vocab_size]
    
    # 计算重建误差
    reconstructed_features = torch.mm(coefficients, vocab_embeddings)
    reconstruction_loss = F.mse_loss(image_features, reconstructed_features).item()
    
    return coefficients, reconstruction_loss

def analyze_patch_similarity(
    model,
    image_processor,
    image_path: str,
    tokenizer,
    output_path: str = None,
    top_k: int = 5,
    vis_ratio: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """
    分析图像patch的词表表示系数，同时比较余弦相似度结果
    
    新增参数:
        output_path: 可视化结果保存路径
        top_k: 显示的top token数量
        vis_ratio: 可视化图像缩放比例
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
    representation_coefficients, loss = compute_least_squares_representation(image_features, vocab_embeddings)
    
    # 计算余弦相似度
    image_features_norm = F.normalize(image_features, dim=1)
    vocab_embeddings_norm = F.normalize(vocab_embeddings, dim=1)
    similarity = torch.mm(image_features_norm, vocab_embeddings_norm.t())
    
    if output_path:
        # 创建两个可视化结果：一个用于最小二乘法，一个用于余弦相似度
        ls_output_path = output_path.replace('.', '_ls.')
        cos_output_path = output_path.replace('.', '_cos.')
        
        # 可视化最小二乘结果
        visualize_patch_tokens(
            image,
            representation_coefficients,
            vocab_embeddings,
            tokenizer,
            ls_output_path,
            top_k=top_k,
            vis_ratio=vis_ratio
        )
        
        # 可视化余弦相似度结果
        visualize_patch_tokens(
            image,
            similarity,
            vocab_embeddings,
            tokenizer,
            cos_output_path,
            top_k=top_k,
            vis_ratio=vis_ratio
        )
    
    # 打印比较结果
    print("\n比较最小二乘法和余弦相似度的结果:")
    for patch_idx in range(representation_coefficients.shape[0]):
        ls_coeffs = representation_coefficients[patch_idx]
        cos_coeffs = similarity[patch_idx]
        
        ls_top_values, ls_top_indices = torch.topk(ls_coeffs, top_k)
        cos_top_values, cos_top_indices = torch.topk(cos_coeffs, top_k)
        
        print(f"\nPatch {patch_idx}:")
        print("最小二乘法 top tokens:")
        for value, idx in zip(ls_top_values, ls_top_indices):
            token = tokenizer.decode([idx])
            print(f"  Token: {token}, 系数: {value:.4f}")
            
        print("余弦相似度 top tokens:")
        for value, idx in zip(cos_top_values, cos_top_indices):
            token = tokenizer.decode([idx])
            print(f"  Token: {token}, 相似度: {value:.4f}")
    
    return representation_coefficients, loss

if __name__ == "__main__":
    import argparse
    from llava.model.builder import load_pretrained_model
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="LLaVA模型路径")
    parser.add_argument("--image-path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output-path", type=str, help="可视化结果保存路径")
    parser.add_argument("--top-k", type=int, default=5, help="显示的top token数量")
    parser.add_argument("--vis-ratio", type=float, default=1.0, help="可视化图像缩放比例")
    args = parser.parse_args()
    
    # 加载模型
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    
    # 计算相似度
    coefficients, loss = analyze_patch_similarity(
        model,
        image_processor,
        args.image_path,
        tokenizer,
        args.output_path,
        args.top_k,
        args.vis_ratio
    )

    