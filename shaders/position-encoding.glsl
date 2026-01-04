#version 450

// Sinusoidal positional encoding for transformer
// Adds position information to token embeddings

layout(local_size_x = 256) in;

layout(binding = 0) buffer Input {
    float embeddings[];
};

layout(binding = 1) buffer Output {
    float encoded[];
};

layout(push_constant) uniform Params {
    uint batch_size;
    uint seq_len;
    uint hidden_dim;
    uint max_position;
} params;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = params.batch_size * params.seq_len * params.hidden_dim;
    
    if (idx >= total) return;
    
    uint batch = idx / (params.seq_len * params.hidden_dim);
    uint pos = (idx / params.hidden_dim) % params.seq_len;
    uint dim = idx % params.hidden_dim;
    
    float position = float(pos);
    float dim_pair = floor(float(dim) / 2.0);
    
    float angle = position / pow(10000.0, (2.0 * dim_pair) / float(params.hidden_dim));
    
    float pos_enc;
    if (dim % 2 == 0) {
        pos_enc = sin(angle);
    } else {
        pos_enc = cos(angle);
    }
    
    encoded[idx] = embeddings[idx] + pos_enc;
}
