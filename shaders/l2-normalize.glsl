#version 450

// L2 normalization for sentence embeddings
// Critical for cosine similarity to work properly

layout(local_size_x = 256) in;

layout(binding = 0) buffer Input {
    float embeddings[];
};

layout(binding = 1) buffer Output {
    float normalized[];
};

layout(push_constant) uniform Params {
    uint batch_size;
    uint hidden_dim;
} params;

shared float shared_norm[256];

void main() {
    uint batch = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    
    float sum_sq = 0.0;
    
    for (uint i = tid; i < params.hidden_dim; i += 256) {
        uint idx = batch * params.hidden_dim + i;
        float val = embeddings[idx];
        sum_sq += val * val;
    }
    
    shared_norm[tid] = sum_sq;
    barrier();
    
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_norm[tid] += shared_norm[tid + stride];
        }
        barrier();
    }
    
    float norm = sqrt(shared_norm[0]) + 1e-8;
    
    for (uint i = tid; i < params.hidden_dim; i += 256) {
        uint idx = batch * params.hidden_dim + i;
        normalized[idx] = embeddings[idx] / norm;
    }
}
