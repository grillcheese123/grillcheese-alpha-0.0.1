#version 450

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Attention weights (batch, num_heads, seq_len, seq_len) - softmax already applied
layout(set = 0, binding = 0) readonly buffer AttentionWeights {
    float weights[];
};

// Values (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 1) readonly buffer Values {
    float V[];
};

// Output (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 2) buffer Output {
    float output_data[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
};

void main() {
    // Each thread computes one output element
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    
    uint total_positions = batch_size * seq_len * num_heads;
    
    if (row >= total_positions || col >= head_dim) {
        return;
    }
    
    // Decode position
    uint batch_idx = row / (seq_len * num_heads);
    uint remainder = row % (seq_len * num_heads);
    uint q_pos = remainder / num_heads;
    uint head_idx = remainder % num_heads;
    
    // Compute weighted sum: output = attention_weights @ V
    float sum = 0.0;
    
    for (uint k_pos = 0; k_pos < seq_len; k_pos++) {
        // Attention weight index
        uint weight_idx = batch_idx * num_heads * seq_len * seq_len +
                         head_idx * seq_len * seq_len +
                         q_pos * seq_len + k_pos;
        
        // Value index
        uint v_idx = batch_idx * seq_len * num_heads * head_dim +
                    k_pos * num_heads * head_dim +
                    head_idx * head_dim + col;
        
        sum += weights[weight_idx] * V[v_idx];
    }
    
    // Output index
    uint out_idx = batch_idx * seq_len * num_heads * head_dim +
                   q_pos * num_heads * head_dim +
                   head_idx * head_dim + col;
    
    output_data[out_idx] = sum;
}
