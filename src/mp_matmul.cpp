#ifndef FORWARD_H
#define FORWARD_H

#include "hls_stream.h" // Needed for hls::stream
#include "config.h"     // Defines constants like dim, GS, etc.
#include "typedefs.h"   // Defines Config, Transformer, QuantizedTensor structs etc. (NOW CORRECTED)
#include <cstring>      // For memcpy, memset
#include "hls_math.h"  // For hls_math functions like sqrtf, expf, cosf, sinf, powf
#include "ap_int.h"
// #include <cmath>        // For fabs, round, sqrtf, expf, cosf, sinf, powf


// ----------------------------------------------------------------------------
// Typedef for the specific TransformerWeights instantiation using config.h constants
// --- UPDATED typedef to pass GS ---
// ----------------------------------------------------------------------------
typedef TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> ModelWeights_t;


// ----------------------------------------------------------------------------
// Template function definitions (kept in header for template instantiation)
// --- UPDATED quantize/dequantize signatures ---
// ----------------------------------------------------------------------------

// --- UPDATED Signature ---
template <int S, int GROUP_SIZE> // Added GROUP_SIZE template parameter
void dequantize(const QuantizedTensor<S, GROUP_SIZE> *qx, float x[S], int group_size) { // Made qx const, Renamed param
    // Check parameter consistency (optional but good)
    if (group_size <= 0 || group_size != GROUP_SIZE) {
        // Handle error: maybe fill x with zeros
        memset(x, 0, S * sizeof(float));
        return;
    }

    for (int i = 0; i < S; i++) {
#pragma HLS PIPELINE II=1
        // Use group_size parameter (which should match GROUP_SIZE template param)
        int scale_idx = i / group_size;
        x[i] = qx->q[i] * qx->s[scale_idx];
    }
}

// --- UPDATED Signature & FIXED hardcoded 64 ---
template <int S, int GROUP_SIZE> // Added GROUP_SIZE template parameter
void quantize_int8(QuantizedTensor<S, GROUP_SIZE> *qx, const float x[S], int group_size) { // made x const, Renamed param

    // Check parameter consistency (optional but good)
    if (group_size <= 0 || group_size != GROUP_SIZE || S % group_size != 0) {
        // Handle error or default behavior
        memset(qx->q, 0, S * sizeof(ap_int<8>));
        int scale_size = (S > 0 && group_size > 0 && S % group_size == 0) ? (S / group_size) : 0;
        if (scale_size > 0) {
            memset(qx->s, 0, scale_size * sizeof(float));
        }
        return;
    }

    constexpr float Q_MAX = 127.0f;
    // Calculate num_groups based on the 'group_size' parameter
    const int num_groups = S / GS; // <<< FIXED: Use parameter

    // Local buffers (Consider stack limits for very large S)
    // Allocate buffer size based on calculated num_groups
    float scale_buffer[num_groups];
    int8_t quantized_buffer[S];

#pragma HLS ARRAY_PARTITION variable = quantized_buffer type = cyclic factor = 64 // Example factor, adjust as needed
#pragma HLS ARRAY_PARTITION variable = scale_buffer type = cyclic factor = 16   // Example factor, adjust as needed

main_loop:
    for (int group = 0; group < num_groups; group++) {
        #pragma HLS UNROLL factor = 4 // Example factor, adjust as needed
        #pragma HLS PIPELINE
        float wmax = 0.0;
        // Use 'group_size' parameter
        int base_idx = group * group_size;

    max_val_loop:
        // Use 'group_size' parameter
        for (int i = 0; i < group_size; i++) {
        #pragma HLS PIPELINE // Inner loop pipeline may or may not be needed/beneficial
            float val = hls::fabs(x[base_idx + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        float scale = (wmax > 1e-9f) ? (wmax / Q_MAX) : 0.0f;
        scale_buffer[group] = scale;
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

    quant_loop:
        // Use 'group_size' parameter
        for (int i = 0; i < group_size; i++) {
        #pragma HLS PIPELINE // Inner loop pipeline may or may not be needed/beneficial
            float quant_value = (scale != 0.0f) ? (x[base_idx + i] * inv_scale) : 0.0f;
            // Clamp using ternary operators or std::min/max if available/synthesizable
            quant_value = (quant_value > Q_MAX) ? Q_MAX : quant_value;
            quant_value = (quant_value < -Q_MAX) ? -Q_MAX : quant_value;
            int8_t quantized = (int8_t)hls::roundf(quant_value);
            quantized_buffer[base_idx + i] = quantized;
        }
    }

    // Copy results to output struct
    std::memcpy(qx->q, quantized_buffer, S * sizeof(int8_t));
    // Copy correct number of scales
    std::memcpy(qx->s, scale_buffer, num_groups * sizeof(float)); // <<< Correct size based on num_groups
}

template <int S, int GROUP_SIZE> // Added GROUP_SIZE template parameter
void quantize_int4(QuantizedTensor<S, GROUP_SIZE> *qx, const float x[S], int group_size) { // made x const, Renamed param

    // Check parameter consistency (optional but good)
    if (group_size <= 0 || group_size != GROUP_SIZE || S % group_size != 0) {
        // Handle error or default behavior
        memset(qx->q, 0, S * sizeof(ap_int<4>));
        int scale_size = (S > 0 && group_size > 0 && S % group_size == 0) ? (S / group_size) : 0;
        if (scale_size > 0) {
            memset(qx->s, 0, scale_size * sizeof(float));
        }
        return;
    }

    constexpr float Q_MAX = 15.0f;
    // Calculate num_groups based on the 'group_size' parameter
    const int num_groups = S / GS; // <<< FIXED: Use parameter

    // Local buffers (Consider stack limits for very large S)
    // Allocate buffer size based on calculated num_groups
    float scale_buffer[num_groups];
    int8_t quantized_buffer[S];

#pragma HLS ARRAY_PARTITION variable = quantized_buffer type = cyclic factor = 64
#pragma HLS ARRAY_PARTITION variable = scale_buffer type = cyclic factor = 16

main_loop:
    for (int group = 0; group < num_groups; group++) {
        #pragma HLS UNROLL factor = 4
        #pragma HLS PIPELINE
        float wmax = 0.0;
        // Use 'group_size' parameter
        int base_idx = group * group_size;

    max_val_loop:
        // Use 'group_size' parameter
        for (int i = 0; i < group_size; i++) {
        #pragma HLS PIPELINE // Inner loop pipeline may or may not be needed/beneficial
            float val = hls::fabs(x[base_idx + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        float scale = (wmax > 1e-9f) ? (wmax / Q_MAX) : 0.0f;
        scale_buffer[group] = scale;
        float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;

    quant_loop:
        // Use 'group_size' parameter
        for (int i = 0; i < group_size; i++) {
        #pragma HLS PIPELINE // Inner loop pipeline may or may not be needed/beneficial
            float quant_value = (scale != 0.0f) ? (x[base_idx + i] * inv_scale) : 0.0f;
            // Clamp using ternary operators or std::min/max if available/synthesizable
            quant_value = (quant_value > Q_MAX) ? Q_MAX : quant_value;
            quant_value = (quant_value < -Q_MAX) ? -Q_MAX : quant_value;
            ap_int<4> quantized = (ap_int<4>)hls::roundf(quant_value);
            quantized_buffer[base_idx + i] = quantized;
        }
    }

    // Copy results to output struct
    std::memcpy(qx->q, quantized_buffer, S * sizeof(ap_int<4>));
    // Copy correct number of scales
    std::memcpy(qx->s, scale_buffer, num_groups * sizeof(float)); // <<< Correct size based on num_groups
}

template <int S>
// const 限定符已在上次修正中添加
void rmsnorm(float o[S], const float x[S], const float weight[S]) {
    constexpr auto array_size = S * sizeof(float);
    float ss = 0.0f;
    float x_buff[S];
    float weight_buff[S];
    float out_buff[S];
// 根据需要调整分区因子
#pragma HLS array_partition variable = x_buff type = cyclic factor = 2
#pragma HLS array_partition variable = weight_buff type = cyclic factor = 2
#pragma HLS array_partition variable = out_buff type = cyclic factor = 2 // 示例因子
    std::memcpy(x_buff, x, array_size);
    std::memcpy(weight_buff, weight, array_size);

sum_of_squares:
    for (int j = 0; j < S; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2 skip_exit_check // 示例因子
        float x_j = x_buff[j];
        ss += x_j * x_j;
    }
    ss /= S;
    ss += 1e-5f;
    ss = 1.0f / hls::sqrtf(ss);

norm_and_scale:
    for (int j = 0; j < S; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2 // 示例因子
        float weight_j = weight_buff[j];
        float x_j = x_buff[j];
        out_buff[j] = weight_j * (ss * x_j);
    }
    std::memcpy(o, out_buff, array_size);
}

// Softmax 定义保持不变 (它不直接依赖于 QuantizedTensor 的改动)
template <int MAXSIZE>
void softmax(float *x, int size) {
    float buffer[MAXSIZE];
#pragma HLS array_partition variable=buffer type=cyclic factor=2 // Example factor
    if (size <= 0) return;
    float max_val = x[0];
max:
    for (int i = 1; i < size; i++) {
#pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2 // Use config constants
#pragma HLS PIPELINE II=1
        if (x[i] > max_val) { max_val = x[i]; }
    }
    float sum = 0.0f;
exp_sum: // Merged loop from previous example version
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor = 2 // Example factor
        buffer[i] = hls::expf(x[i] - max_val);
        sum += buffer[i];
    }
    const float inv_sum = (sum == 0.0f) ? 0.0f : 1.0f / sum;
norm:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 1 max = seq_len avg = seq_len/2
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor = 2 // Example factor
        x[i] = buffer[i] * inv_sum;
    }
}

const int MATMUL_UNROLL_FACTOR = 16;
template <int N, int D>
// const 限定符已在上次修正中添加
// 此函数内部逻辑依赖于全局 GS 常量，并处理从 QuantizedTensor 传入的原始指针
// 功能上与修正后的 QuantizedTensor<SIZE, GS> 内存布局兼容
void matmul_w8a8(float *xout, const ap_int<8> *xq, const float *xs, const ap_int<8> *wq, const float *ws) {
    // W (d,n) @ x (n,) -> xout (d,)

    // 假设 GS 是通过 config.h 可用的全局常量

    ap_int<8> x_buffer[N];
    // 大小基于全局 GS
    float xs_buffer[N / GS];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor

x_buff:
    for (int i = 0; i < N; i++) {
    #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR // Example factor
        x_buffer[i] = xq[i];
    }
xs_buff:
     // 加载对应分组的 scale 因子
     for (int j = 0; j < N / GS; j++) { // Loop over groups
         #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR // Example factor
         xs_buffer[j] = xs[j]; // Assumes xs directly corresponds to groups
     }


    for (int i = 0; i < D; i++) { // Loop over output dimension
        // #pragma HLS PIPELINE II=1
        float val = 0.0f;
        ap_int<8> w_buffer[N];
        // 大小基于全局 GS
        float ws_buffer[N / GS];
        #pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor
        #pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor

        const int in_w = i * N;       // Start index in wq for row i
        const int in_s = i * (N / GS); // Start index in ws for row i

    load_w: // Load weights for current row
        for (int j = 0; j < N; j++) {
        #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR // Consider full unroll if N is small enough, or partial
            w_buffer[j] = wq[j + in_w];
        }
    load_ws: // Load scales for current row
        for (int j = 0; j < N / GS; j++) { // Loop over groups
        #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR // Consider full unroll if N/GS is small enough, or partial
            ws_buffer[j] = ws[j + in_s]; // Assumes ws directly corresponds to groups
        }

        // --- 使用之前重写的计算逻辑 (似乎更适合 HLS) ---
        // Perform dot product using groups
        int32_t group_sum[N/GS];
        #pragma HLS ARRAY_PARTITION variable=group_sum complete // Partition for parallel accumulation

    dot_product_groups:
        for (int j = 0; j < N / GS; ++j) { // Loop over groups
        #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll group calculation
            int32_t ival = 0;
        inner_dot:
            for(int k=0; k<GS; ++k) { // Loop within group
            #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll inner dot product
                // Use static buffers loaded earlier
                ival += ((int16_t)x_buffer[j*GS + k]) * ((int16_t)w_buffer[j*GS + k]);
            }
            group_sum[j] = ival; // Store sum for the group
        }

    final_sum: // Accumulate scaled group results
        for(int j=0; j<N/GS; ++j) { // Loop over groups
        #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll final summation
             // Use loaded scales
            val += ((float)group_sum[j]) * ws_buffer[j] * xs_buffer[j];
        }
        // --------------------------------------------------

        xout[i] = val; // Store final output value
    }
}

template <int N, int D>
void matmul_w4a4(float *xout, const ap_int<4> *xq, const float *xs, const ap_int<4> *wq, const float *ws) {
    // W (d,n) @ x (n,) -> xout (d,)

    // 假设 GS 是通过 config.h 可用的全局常量

    ap_int<4> x_buffer[N];
    // 大小基于全局 GS
    float xs_buffer[N / GS];

#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor

x_buff:
    for (int i = 0; i < N; i++) {
    #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR // Example factor
        x_buffer[i] = xq[i];
    }
xs_buff:
     // 加载对应分组的 scale 因子
     for (int j = 0; j < N / GS; j++) { // Loop over groups
         #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR // Example factor
         xs_buffer[j] = xs[j]; // Assumes xs directly corresponds to groups
     }


    for (int i = 0; i < D; i++) { // Loop over output dimension
        // #pragma HLS PIPELINE II=1
        float val = 0.0f;
        ap_int<4> w_buffer[N];
        // 大小基于全局 GS
        float ws_buffer[N / GS];
        #pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor
        #pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor

        const int in_w = i * N;       // Start index in wq for row i
        const int in_s = i * (N / GS); // Start index in ws for row i

    load_w: // Load weights for current row
        for (int j = 0; j < N; j++) {
        #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR // Consider full unroll if N is small enough, or partial
            w_buffer[j] = wq[j + in_w];
        }
    load_ws: // Load scales for current row
        for (int j = 0; j < N / GS; j++) { // Loop over groups
        #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR // Consider full unroll if N/GS is small enough, or partial
            ws_buffer[j] = ws[j + in_s]; // Assumes ws directly corresponds to groups
        }

        // --- 使用之前重写的计算逻辑 (似乎更适合 HLS) ---
        // Perform dot product using groups
        int32_t group_sum[N/GS];
        #pragma HLS ARRAY_PARTITION variable=group_sum complete // Partition for parallel accumulation

    dot_product_groups:
        for (int j = 0; j < N / GS; ++j) { // Loop over groups
        #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll group calculation
            int32_t ival = 0;
        inner_dot:
            for(int k=0; k<GS; ++k) { // Loop within group
            #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll inner dot product
                // Use static buffers loaded earlier
                ival += ((int16_t)x_buffer[j*GS + k]) * ((int16_t)w_buffer[j*GS + k]);
            }
            group_sum[j] = ival; // Store sum for the group
        }

    final_sum: // Accumulate scaled group results
        for(int j=0; j<N/GS; ++j) { // Loop over groups
        #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll final summation
             // Use loaded scales
            val += ((float)group_sum[j]) * ws_buffer[j] * xs_buffer[j];
        }
        // --------------------------------------------------

        xout[i] = val; // Store final output value
    }
}

template <int N, int D>
void matmul_w8a4(float *xout, 
    const ap_int<8> *xq, const float *xs, 
    const ap_int<4> *wq, const float *ws) {
    // TODO
}

template <int N, int D>
void matmul_w4a8(float *xout, 
    const ap_int<4> *xq, const float *xs, 
    const ap_int<8> *wq, const float *ws) {
    // TODO
}
#endif // FORWARD_H