#include "config.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include "ap_float.h"
#include "hls_math.h"
// #include <cmath>     // For sqrtf, expf, cosf, sinf, powf
#include <cstring>

#define N 768
#define D 32000
#define GS 64

typedef ap_int<8> xq_type;
typedef ap_int<8> wq_type;

typedef float fp_type;

const int MATMUL_UNROLL_FACTOR = 2;

extern void mac2dsp(ap_int<8> a0, ap_int<8> a1, ap_int<8> b0, ap_int<32> &p0, ap_int<32> &p1);

void matmul_dsp(
    fp_type *xout, 
    const xq_type *xq, 
    const fp_type *xs, 
    const wq_type *wq, 
    const fp_type *ws) {

    xq_type x_buffer[N];
    fp_type xs_buffer[N / GS];
    #pragma HLS PIPELINE off
    x_buff:
        for (int i = 0; i < N; i++) {
            x_buffer[i] = xq[i];
        }

    xs_buff:
        for (int j = 0; j < N / GS; j++) {
            xs_buffer[j] = xs[j];
        }

    row_pair_loop:
    for (int r = 0; r < D; r += 2) {
        fp_type val0 = (fp_type)0.0f;
        fp_type val1 = (fp_type)0.0f;

        const int base_w0 = r * N;
        const int base_w1 = (r + 1) * N;
        const int base_s0 = r * (N / GS);
        const int base_s1 = (r + 1) * (N / GS);

        group_loop:
        for (int g = 0; g < N / GS; ++g) {
            ap_int<32> p0 = 0;
            ap_int<32> p1 = 0;

            const int g_off = g * GS;

            k_loop:
            for (int k = 0; k < GS; ++k) {
                #pragma HLS UNROLL factor = 8
                const int idx = g_off + k;

                ap_int<8> b0 = x_buffer[idx];                 // shared x
                ap_int<8> a0 = wq[base_w0 + idx];             // row r
                ap_int<8> a1 = wq[base_w1 + idx];             // row r+1

                mac2dsp(a0, a1, b0, p0, p1);
            }

            const fp_type sx  = xs_buffer[g];
            const fp_type sw0 = ws[base_s0 + g];
            const fp_type sw1 = ws[base_s1 + g];

            val0 += (fp_type)((int)p0) * sx * sw0;
            val1 += (fp_type)((int)p1) * sx * sw1;
        }
        xout[r]     = val0;
        xout[r + 1] = val1;
    }
}

void matmul(fp_type *xout, const xq_type *xq, const fp_type *xs, const wq_type *wq, const fp_type *ws) {
    // W (d,n) @ x (n,) -> xout (d,)

    // 假设 GS 是通过 config.h 可用的全局常量

    xq_type x_buffer[N];
    // 大小基于全局 GS
    fp_type xs_buffer[N / GS];

// #pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor
// #pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor

    x_buff:
    for (int i = 0; i < N; i++) {
        // #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Example factor
        x_buffer[i] = xq[i];
    }
    xs_buff:
     // 加载对应分组的 scale 因子
     for (int j = 0; j < N / GS; j++) { // Loop over groups
        // #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Example factor
         xs_buffer[j] = xs[j]; // Assumes xs directly corresponds to groups
     }


    for (int i = 0; i < D; i++) { // Loop over output dimension
        // #pragma HLS PIPELINE II=1
        fp_type val = (fp_type)0;
        wq_type w_buffer[N];
        // 大小基于全局 GS
        fp_type ws_buffer[N / GS];
        // #pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor
        // #pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = MATMUL_UNROLL_FACTOR // Example factor

        const int in_w = i * N;       // Start index in wq for row i
        const int in_s = i * (N / GS); // Start index in ws for row i

    load_w: // Load weights for current row
        for (int j = 0; j < N; j++) {
        // #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Consider full unroll if N is small enough, or partial
            w_buffer[j] = wq[j + in_w];
        }
    load_ws: // Load scales for current row
        for (int j = 0; j < N / GS; j++) { // Loop over groups
        // #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Consider full unroll if N/GS is small enough, or partial
            ws_buffer[j] = ws[j + in_s]; // Assumes ws directly corresponds to groups
        }

        // Perform dot product using groups
        int32_t group_sum[N/GS];
        // #pragma HLS ARRAY_PARTITION variable=group_sum complete // Partition for parallel accumulation

    dot_product_groups:
        for (int j = 0; j < N / GS; ++j) { // Loop over groups
            // #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll group calculation
            int32_t ival = 0;
            int offset = j * GS;
        inner_dot:
            for(int k=0; k<GS; ++k) { // Loop within group
            // #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll inner dot product
                #pragma HLS UNROLL factor = 8
                // Use static buffers loaded earlier
                int addr = offset + k;
                ival += (x_buffer[addr]) * (w_buffer[addr]);
            }
            group_sum[j] = ival; // Store sum for the group
        }

    final_sum: // Accumulate scaled group results
        for(int j=0; j<N/GS; ++j) { // Loop over groups
        // #pragma HLS UNROLL factor = MATMUL_UNROLL_FACTOR// Unroll final summation
             // Use loaded scales
            val += ((fp_type)group_sum[j]) * ws_buffer[j] * xs_buffer[j];
        }
        // --------------------------------------------------

        xout[i] = val; // Store final output value
    }
}

#define S 768
#define GROUP_SIZE 64

void quantize(QuantizedTensor<S, GROUP_SIZE> *qx, const float x[S], int group_size) { // made x const, Renamed param

    // Check parameter consistency (optional but good)
    if (group_size <= 0 || group_size != GROUP_SIZE || S % group_size != 0) {
        // Handle error or default behavior
        memset(qx->q, 0, S * sizeof(int8_t));
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
        float inv_scale = (scale != 0.0f) ? hls::recipf(scale) : 0.0f;

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