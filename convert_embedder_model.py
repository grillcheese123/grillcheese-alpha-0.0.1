"""
Convert sentence-transformers model to Vulkan embedder format
"""

import numpy as np
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import json

def convert_sentence_transformer(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: Path = Path("models")
):
    """Convert a sentence-transformers model to Vulkan format"""
    print(f"Loading model: {model_name}")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    except ImportError:
        print("ERROR: sentence-transformers not installed")
        print("Install: pip install sentence-transformers")
        return None, None
    
    transformer = model[0].auto_model
    config = transformer.config
    
    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")
    
    weights = {}
    
    embedding_table = transformer.embeddings.word_embeddings.weight.data.cpu().numpy()
    weights['embedding_table'] = embedding_table.astype(np.float32)
    print(f"\n✓ Embedding table: {embedding_table.shape}")
    
    for layer_idx in range(config.num_hidden_layers):
        layer = transformer.encoder.layer[layer_idx]
        
        q_weight = layer.attention.self.query.weight.data.cpu().numpy()
        k_weight = layer.attention.self.key.weight.data.cpu().numpy()
        v_weight = layer.attention.self.value.weight.data.cpu().numpy()
        
        qkv_weight = np.concatenate([q_weight, k_weight, v_weight], axis=0)
        weights[f'layer_{layer_idx}_attn_qkv'] = qkv_weight.T.astype(np.float32)
        
        ffn_fc1 = layer.intermediate.dense.weight.data.cpu().numpy()
        weights[f'layer_{layer_idx}_ffn_fc1'] = ffn_fc1.T.astype(np.float32)
        
        ffn_fc2 = layer.output.dense.weight.data.cpu().numpy()
        weights[f'layer_{layer_idx}_ffn_fc2'] = ffn_fc2.T.astype(np.float32)
        
        print(f"✓ Layer {layer_idx}")
    
    output_dir.mkdir(exist_ok=True)
    weights_path = output_dir / "embedder_weights.npz"
    np.savez(weights_path, **weights)
    print(f"\n✓ Saved weights to {weights_path}")
    
    config_dict = {
        'vocab_size': config.vocab_size,
        'hidden_dim': config.hidden_size,
        'num_layers': config.num_hidden_layers,
        'num_heads': config.num_attention_heads,
        'ffn_dim': config.intermediate_size,
        'max_seq_len': config.max_position_embeddings
    }
    
    config_path = output_dir / "embedder_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ Saved config to {config_path}")
    
    return weights_path, config_path

if __name__ == "__main__":
    print("="*60)
    print("Converting sentence-transformers to Vulkan format")
    print("="*60)
    
    weights_path, config_path = convert_sentence_transformer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        output_dir=Path("models")
    )
    
    if weights_path:
        print("\n" + "="*60)
        print("Success! Next steps:")
        print("="*60)
        print("1. Compile shaders:")
        print("   cd shaders")
        print("   glslangValidator -V position-encoding.glsl -o spv/position-encoding.spv")
        print("   glslangValidator -V mean-pooling.glsl -o spv/mean-pooling.spv")
        print("   glslangValidator -V l2-normalize.glsl -o spv/l2-normalize.spv")
        print("\n2. Test embedder:")
        print("   python test_vulkan_embedder.py")
