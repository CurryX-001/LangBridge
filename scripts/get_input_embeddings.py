#!/usr/bin/env python3
"""
Extract and process input embeddings from pretrained models.

This script provides functionality to:
1. Extract embedding matrices from pretrained models
2. Process embeddings according to vocabulary mappings
"""

import torch
from transformers import AutoModel
import json
import argparse
import os
from pathlib import Path


def get_model_embeddings(model_name: str, save_path: str) -> None:
    """
    Extract and save embedding matrix from pretrained model.
    
    Args:
        model_name: HuggingFace model name or path
        save_path: Path to save the embedding matrix
    """
    print(f"Loading model: {model_name}")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    embedding_matrix = model.get_input_embeddings().weight
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(embedding_matrix, save_path)
    print(f"Embedding matrix saved to: {save_path}")
    print(f"Shape: {embedding_matrix.shape}")


def process_embeddings(embedding_path: str, vocab_path: str, save_path: str) -> None:
    """
    Process embeddings according to vocabulary mapping.
    
    Args:
        embedding_path: Path to embedding matrix file
        vocab_path: Path to vocabulary JSON file
        save_path: Path to save processed embeddings
    """
    print(f"Loading embedding matrix from: {embedding_path}")
    embedding_matrix = torch.load(embedding_path, map_location='cpu')
    if hasattr(embedding_matrix, 'weight'):
        embedding_matrix = embedding_matrix.weight
    
    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as file:
        vocab_data = json.load(file)
    
    # Validate indices
    indices = list(vocab_data.values())
    max_index = embedding_matrix.size(0)
    if max(indices) >= max_index:
        raise ValueError(f"Vocabulary index out of range. Max allowed: {max_index-1}")
    
    # Select corresponding embeddings
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    mm_vocab = torch.index_select(embedding_matrix, 0, indices_tensor)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(mm_vocab, save_path)
    print(f"Processed embeddings saved to: {save_path}")
    print(f"Vocabulary size: {len(vocab_data)}")
    print(f"Embedding shape: {mm_vocab.shape}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Extract and process model embeddings"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--vocab_path", 
        type=str,
        required=True,
        help="Path to vocabulary JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./embeddings",
        help="Output directory for embeddings"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate file paths
    model_safe_name = args.model_name.replace("/", "_")
    vocab_name = Path(args.vocab_path).stem
    
    embedding_path = output_dir / f"{model_safe_name}_embeddings.pt"
    processed_path = output_dir / f"{model_safe_name}_{vocab_name}_processed.pt"
    
    # Step 1: Extract embeddings
    print("Step 1: Extracting model embeddings...")
    get_model_embeddings(args.model_name, str(embedding_path))
    
    # Step 2: Process embeddings with vocabulary
    print("Step 2: Processing embeddings with vocabulary...")
    process_embeddings(str(embedding_path), args.vocab_path, str(processed_path))
    
    print("Embedding extraction completed successfully!")


if __name__ == "__main__":
    main()