#!/usr/bin/env python3
"""
Create vocabulary subsets by taking the first N tokens from the base vocabulary file.
Usage: python scripts/create_vocab_subset.py --vocab_size 19200 --model_name llama3
"""

import json
import argparse
from pathlib import Path

def create_vocab_subset(base_vocab_path, vocab_size, model_name, output_dir="vocab"):
    """Create a vocabulary subset by taking the first N tokens."""
    
    # Load base vocabulary
    with open(base_vocab_path, 'r') as f:
        base_vocab = json.load(f)
    
    # Take first N tokens (tokens are already sorted by frequency in the base file)
    tokens = list(base_vocab.keys())[:vocab_size]
    subset_vocab = {token: base_vocab[token] for token in tokens}
    
    # Create output filename
    output_path = Path(output_dir) / f"{vocab_size}_{model_name}_sub_llava_share_intersect_llama_qwen.json"
    output_path.parent.mkdir(exist_ok=True)
    
    # Save subset vocabulary
    with open(output_path, 'w') as f:
        json.dump(subset_vocab, f, indent=2)
    
    print(f"Created vocabulary subset: {output_path}")
    print(f"Original vocab size: {len(base_vocab)}")
    print(f"Subset vocab size: {len(subset_vocab)}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create vocabulary subsets")
    parser.add_argument("--vocab_size", type=int, required=True, 
                       help="Target vocabulary size (e.g., 19200, 25600, 32000)")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Model name (e.g., llama3, Qwen)")
    parser.add_argument("--base_vocab", type=str, 
                       default="vocab/sub_llava_share_intersect_llama_qwen.json",
                       help="Path to base vocabulary file")
    parser.add_argument("--output_dir", type=str, default="vocab",
                       help="Output directory for vocabulary files")
    
    args = parser.parse_args()
    
    create_vocab_subset(args.base_vocab, args.vocab_size, args.model_name, args.output_dir)

if __name__ == "__main__":
    main()