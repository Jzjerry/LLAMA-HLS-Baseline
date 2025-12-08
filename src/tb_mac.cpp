/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
#include "config.h"   // Defines model parameters (dim, hidden_dim, etc.) - Provides global GS
#include "forward.h"  // Declares kernel functions, helper templates (quantize/dequantize), and types
#include "typedefs.h" // Contains corrected type definitions (QuantizedTensor<SIZE, GROUP_SIZE>, etc.)
#include <cstring>
#include <ctype.h>
#include <fcntl.h>
#include <iostream>
#include <math.h>
#include <hls_math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include "hls_stream.h" // HLS Stream header
#include <string>
#include <cstdlib> // 关键：包含这个头文件以使用 getenv
#include "ap_int.h"
// No XRT includes for csim
#if defined _WIN32
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// ----------------------------------------------------------------------------
// --- UPDATED main function ---
// Uses global GS from config.h when instantiating/calling templates
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {

    ap_int<8> test_in[4] = {-1, 1};
    ap_int<8> test_w[4] = {32, -32};

    int32_t result = 0;
    // Reference calculation
    for (int i = 0; i < 2; i++) {
        result += (int32_t)(test_in[i]) * (int32_t)(test_w[i]);
    }

    printf("Reference result = %d\n", result);

    ap_int<27> a = 0;
    a(26, 18) = (ap_int<9>)test_in[0];
    ap_int<27> d = (ap_int<27>)test_in[1];
    ap_int<18> b = (ap_int<18>)test_w[0];

    ap_int<44> prod;
    prod = (a+d)*b;

    ap_int<18> prod1 = prod(17, 0);
    ap_int<18> prod2 = prod(35, 18) + prod(17,17);

    printf("Prod 1 = %d, Prod 2 = %d\n", (int32_t)prod1, (int32_t)prod2);
 
    return 0;
}
