#ifndef FORWARD_H
#define FORWARD_H

// #include "hls_stream.h" // Needed for hls::stream
#include "config.h"     // Defines constants like dim, GS, etc.
#include "typedefs.h"   // Defines Config, Transformer, QuantizedTensor structs etc. (NOW CORRECTED)
#include <cstring>      // For memcpy, memset
#include <cmath>        // For fabs, round, sqrtf, expf, cosf, sinf, powf


// ----------------------------------------------------------------------------
// Typedef for the specific TransformerWeights instantiation using config.h constants
// --- UPDATED typedef to pass GS ---
// ----------------------------------------------------------------------------
typedef TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> ModelWeights_t;

// ----------------------------------------------------------------------------
// Declarations for the split kernels (No changes needed here)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Template function definitions (kept in header for template instantiation)
// --- UPDATED quantize/dequantize signatures ---
// ----------------------------------------------------------------------------

// --- UPDATED Signature ---
template <int S, int GROUP_SIZE> // Added GROUP_SIZE template parameter
void dequantize(const QuantizedTensor<int8_t, S, GROUP_SIZE> *qx, float x[S], int group_size) { // Made qx const, Renamed param
    // Check parameter consistency (optional but good)
    if (group_size <= 0 || group_size != GROUP_SIZE) {
        // Handle error: maybe fill x with zeros
        memset(x, 0, S * sizeof(float));
        return;
    }

    for (int i = 0; i < S; i++) {
        // Use group_size parameter (which should match GROUP_SIZE template param)
        int scale_idx = i / group_size;
        x[i] = qx->q[i] * qx->s[scale_idx];
    }
}

typedef union {
    float f32;
    uint32_t u32;
} float_uint32_union;

// 快速转换函数
inline uint32_t float_to_uint32(float a) {
    float_uint32_union u;
    u.f32 = a;
    return u.u32;
}

inline float uint32_to_float(uint32_t a) {
    float_uint32_union u;
    u.u32 = a;
    return u.f32;
}
#endif