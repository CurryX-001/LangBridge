import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Optional, Tuple, List
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import os

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
    
    # 计算相似度并获取top-k tokens
    image_features_norm = F.normalize(image_features, dim=1)
    vocab_embeddings_norm = F.normalize(vocab_embeddings, dim=1)
    similarity = torch.mm(image_features_norm, vocab_embeddings_norm.t())
    top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=1)
    
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

def analyze_image_patches(
    model,
    image_processor,
    tokenizer,
    image_path: str,
    output_path: str,
    top_k: int = 1,
    vis_ratio: float = 1.0,
) -> None:
    """
    分析图像的patch-token对应关系并生成可视化结果
    """
    # 获取处理后的图像
    image = Image.open(image_path).convert('RGB')
    processed_tensor = process_images([image], image_processor, model.config)[0]

    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


    expanded_image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
    
    # 将处理后的tensor转换回PIL图像用于可视化
    
    with torch.inference_mode():
        image_features = model.get_vision_tower()(
            processed_tensor.unsqueeze(0).half().cuda()
        )
        image_features = model.get_model().mm_projector(image_features)
        image_features = image_features.squeeze(0)
    
    vocab_embeddings = model.get_model().embed_tokens.weight.to(device=image_features.device)
    
    # 使用处理后的图像进行可视化
    visualize_patch_tokens(
        expanded_image,
        image_features,
        vocab_embeddings,
        tokenizer,
        output_path,
        top_k=top_k,
        vis_ratio=vis_ratio
    )

if __name__ == "__main__":
    import argparse
    from llava.model.builder import load_pretrained_model
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="LLaVA模型路径")
    parser.add_argument("--image-path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output-path", type=str, required=True, help="输出图像路径")
    parser.add_argument("--top-k", type=int, default=1, help="每个patch显示的token数量")
    parser.add_argument("--font-size", type=int, default=10, help="文字大小")
    parser.add_argument("--vis-ratio", type=float, default=1.0, help="可视化图像缩放比例")
    args = parser.parse_args()
    
    # 加载模型
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    
    # 分析图像
    analyze_image_patches(
        model,
        image_processor,
        tokenizer,
        args.image_path,
        args.output_path,
        args.top_k,
        args.vis_ratio
    ) 