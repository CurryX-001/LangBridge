# """
# This is just a utility that I use to extract the projector for quantized models.
# It is NOT necessary at all to train, or run inference/serve demos.
# Use this script ONLY if you fully understand its implications.
# """


# import os
# import argparse
# import torch
# import json
# from collections import defaultdict


# def parse_args():
#     parser = argparse.ArgumentParser(description='Extract MMProjector weights')
#     parser.add_argument('--model-path', type=str, help='model folder')
#     parser.add_argument('--output', type=str, help='output file')
#     args = parser.parse_args()
#     return args


# if __name__ == '__main__':
#     args = parse_args()

#     keys_to_match = ['mm_projector']
#     ckpt_to_key = defaultdict(list)
#     try:
#         model_indices = json.load(open(os.path.join(args.model_path, 'pytorch_model.bin.index.json')))
#         for k, v in model_indices['weight_map'].items():
#             if any(key_match in k for key_match in keys_to_match):
#                 ckpt_to_key[v].append(k)
#     except FileNotFoundError:
#         # Smaller models or model checkpoints saved by DeepSpeed.
#         v = 'pytorch_model.bin'
#         for k in torch.load(os.path.join(args.model_path, v), map_location='cpu').keys():
#             if any(key_match in k for key_match in keys_to_match):
#                 ckpt_to_key[v].append(k)

#     loaded_weights = {}

#     for ckpt_name, weight_keys in ckpt_to_key.items():
#         ckpt = torch.load(os.path.join(args.model_path, ckpt_name), map_location='cpu')
#         for k in weight_keys:
#             loaded_weights[k] = ckpt[k]

#     torch.save(loaded_weights, args.output)
from safetensors.torch import load_file
import torch

model_path = "/mnt/data/jiaqi.liao/Codebook/checkpoints/qwen2_0.5b_it_vocab_12800_dim_1024_pretrain_2e-5_finetune/model.safetensors"
state_dict = load_file(model_path)

mm_projector_weights = {}
for key, value in state_dict.items():
    if "mm_projector" in key:
        mm_projector_weights[key] = value

output_path = "/mnt/data/jiaqi.liao/Codebook/qwen2_0.5b_it_vocab_12800_dim_1024_finetune_mm_projector.bin"
torch.save(mm_projector_weights, output_path)

print(f"mm_projector weights have been extracted and saved to {output_path}")