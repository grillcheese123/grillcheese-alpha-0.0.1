#version 450

// Mean pooling over sequence length to get sentence embedding
// Takes attention mask into account (only pool over real tokens)

layout(local_size_x = 256) in;

layout(binding = 0) buffer Input {
    float embeddings[];
};

layout(binding = 1) buffer AttentionMask {
    uint mask[];
};

layout(binding = 2) buffer Output {
    float pooled[];
};

layout(push_constant) uniform Params {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
} params;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    
    if (idx >= params.batch_size * params.hidden_dim) return;
    
    uint batch = idx / params.hidden_dim;
    uint dim = idx % params.hidden_dim;
    
    float sum = 0.0;
    uint count = 0;
    
    for (uint i = 0; i < params.seq_len; i++) {
        uint mask_idx = batch * params.seq_len + i;
        
        if (mask[mask_idx] == 1) {
            uint emb_idx = batch * params.seq_len * params.hidden_dim + 
                          i * params.hidden_dim + dim;
            sum += embeddings[emb_idx];
            count++;
        }
    }
    
    pooled[idx] = (count > 0) ? (sum / float(count)) : 0.0;
}
