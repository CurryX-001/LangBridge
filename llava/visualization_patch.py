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
    Visualize the most similar tokens for each patch directly on the original image
    
    Args:
        image: Original PIL image
        image_features: Visual features [num_patches, hidden_dim]
        vocab_embeddings: Vocabulary embeddings [vocab_size, hidden_dim]
        tokenizer: Tokenizer
        output_path: Output image path
        top_k: Number of tokens to display for each patch
        font_size: Font size
        text_color: Text color
        outline_color: Text outline color
        vis_ratio: Scaling ratio for visualization image
    
    Returns:
        Image annotated with tokens
    """
    # Resize image while maintaining aspect ratio
    new_width = int(image.width * vis_ratio)
    new_height = int(image.height * vis_ratio)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Ensure data type consistency
    vocab_embeddings = vocab_embeddings.to(dtype=image_features.dtype)
    
    # Calculate patch size and grid dimensions
    num_patches = image_features.shape[0]
    grid_size = int(np.sqrt(num_patches))
    patch_width = new_width // grid_size
    patch_height = new_height // grid_size
    
    # Calculate similarity and get top-k tokens
    image_features_norm = F.normalize(image_features, dim=1)
    vocab_embeddings_norm = F.normalize(vocab_embeddings, dim=1)
    similarity = torch.mm(image_features_norm, vocab_embeddings_norm.t())
    top_k_values, top_k_indices = torch.topk(similarity, k=top_k, dim=1)
    
    # Create drawable image copy
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # Try to load font, use default font if failed
    try:
        # For Linux systems
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            # For Windows systems
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Draw corresponding tokens on each patch
    for i in range(num_patches):
        row = i // grid_size
        col = i % grid_size
        x = col * patch_width
        y = row * patch_height
        
        # Get top-k tokens for current patch
        tokens = []
        for idx in top_k_indices[i]:
            token = tokenizer.decode(idx.item()).strip()
            tokens.append(token)
        
        # Merge tokens into a single string
        text = "\n".join(tokens)
        
        # Calculate text bounding box
        bbox = draw.textbbox((x, y), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate centered text position
        text_x = x + (patch_width - text_width) // 2
        text_y = y + (patch_height - text_height) // 2
        
        # Draw text outline
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((text_x+dx, text_y+dy), text, font=font, fill=outline_color)
        
        # Draw text
        draw.text((text_x, text_y), text, font=font, fill=text_color)
        
        # Optional: draw patch boundaries
        draw.rectangle([x, y, x+patch_width, y+patch_height], outline="red", width=1)
    
    # Save result
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
    Analyze patch-token correspondence of image and generate visualization results
    """
    # Get processed image
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
    
    # Convert processed tensor back to PIL image for visualization
    
    with torch.inference_mode():
        image_features = model.get_vision_tower()(
            processed_tensor.unsqueeze(0).half().cuda()
        )
        image_features = model.get_model().mm_projector(image_features)
        image_features = image_features.squeeze(0)
    
    vocab_embeddings = model.get_model().embed_tokens.weight.to(device=image_features.device)
    
    # Use processed image for visualization
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
    parser.add_argument("--model-path", type=str, required=True, help="LLaVA model path")
    parser.add_argument("--image-path", type=str, required=True, help="Input image path")
    parser.add_argument("--output-path", type=str, required=True, help="Output image path")
    parser.add_argument("--top-k", type=int, default=1, help="Number of tokens to display for each patch")
    parser.add_argument("--font-size", type=int, default=10, help="Font size")
    parser.add_argument("--vis-ratio", type=float, default=1.0, help="Visualization image scaling ratio")
    args = parser.parse_args()
    
    # Load model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    
    # Analyze image
    analyze_image_patches(
        model,
        image_processor,
        tokenizer,
        args.image_path,
        args.output_path,
        args.top_k,
        args.vis_ratio
    ) 