#version 450

// RoPE (Rotary Position Embeddings) Shader
// Applies rotary embeddings to Q and K before attention
// Based on: https://arxiv.org/abs/2104.09864 (RoFormer)
//
// RoPE encodes position by rotating pairs of dimensions:
// q'[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
// q'[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)
// where θ = pos * base^(-2i/head_dim)

layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input Q or K (batch, seq_len, num_heads, head_dim)
layout(set = 0, binding = 0) readonly buffer Input {
    float input_data[];
};

// Output rotated Q or K (same shape)
layout(set = 0, binding = 1) buffer Output {
    float output_data[];
};

// Precomputed cos/sin tables (optional - can compute on-the-fly)
// Shape: (max_seq_len, head_dim/2) for cos, same for sin
layout(set = 0, binding = 2) readonly buffer CosTable {
    float cos_table[];
};

layout(set = 0, binding = 3) readonly buffer SinTable {
    float sin_table[];
};

// Parameters
layout(push_constant) uniform PushConsts {
    uint batch_size;
    uint seq_len;
    uint num_heads;
    uint head_dim;
    float rope_base;       // Base for frequency computation (default: 10000.0)
    uint use_precomputed;  // 1 = use tables, 0 = compute on-the-fly
    float rope_scaling;    // For extended context (default: 1.0)
};

// Compute RoPE frequency for dimension pair i at position pos
// θ = pos * base^(-2i/head_dim)
float compute_theta(uint pos, uint dim_pair_idx, float base, uint head_dim_val, float scaling) {
    float position = float(pos) / scaling;
    float freq_exp = -2.0 * float(dim_pair_idx) / float(head_dim_val);
    float freq = pow(base, freq_exp);
    return position * freq;
}

void main() {
    uint gID = gl_GlobalInvocationID.x;
    
    // Total elements = batch * seq * heads * head_dim
    uint total_elements = batch_size * seq_len * num_heads * head_dim;
    
    // Each thread handles one element, but we process pairs
    // Only process even indices to avoid double-processing
    if (gID >= total_elements) return;
    
    // Decode position
    uint batch_idx = gID / (seq_len * num_heads * head_dim);
    uint remainder = gID % (seq_len * num_heads * head_dim);
    uint seq_idx = remainder / (num_heads * head_dim);
    remainder = remainder % (num_heads * head_dim);
    uint head_idx = remainder / head_dim;
    uint dim_idx = remainder % head_dim;
    
    // Only process even dimension indices (we handle pairs)
    if (dim_idx % 2 != 0) {
        return;
    }
    
    uint dim_pair_idx = dim_idx / 2;  // Which pair (0, 1, 2, ...)
    
    // Get input values for this pair
    uint idx_even = gID;  // Even index (already at gID)
    uint idx_odd = gID + 1;  // Odd index (next element)
    
    float x_even = input_data[idx_even];
    float x_odd = input_data[idx_odd];
    
    // Get or compute cos/sin
    float cos_val, sin_val;
    
    if (use_precomputed == 1) {
        // Use precomputed tables
        uint table_idx = seq_idx * (head_dim / 2) + dim_pair_idx;
        cos_val = cos_table[table_idx];
        sin_val = sin_table[table_idx];
    } else {
        // Compute on-the-fly
        float theta = compute_theta(seq_idx, dim_pair_idx, rope_base, head_dim, rope_scaling);
        cos_val = cos(theta);
        sin_val = sin(theta);
    }
    
    // Apply rotation
    // x'[2i]   = x[2i] * cos(θ) - x[2i+1] * sin(θ)
    // x'[2i+1] = x[2i] * sin(θ) + x[2i+1] * cos(θ)
    float rotated_even = x_even * cos_val - x_odd * sin_val;
    float rotated_odd = x_even * sin_val + x_odd * cos_val;
    
    output_data[idx_even] = rotated_even;
    output_data[idx_odd] = rotated_odd;
}
