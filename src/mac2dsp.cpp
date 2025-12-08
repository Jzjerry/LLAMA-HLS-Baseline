#include "ap_int.h"

void mac2dsp(
    ap_int<8> a0, ap_int<8> a1, ap_int<8> b0, 
    ap_int<32> &p0, ap_int<32> &p1) {
    
    // p0 += a0 * b0;
    // p1 += a1 * b0;

    // Shared DSP implementation
    
    ap_int<27> a = 0;
    a(26, 18) = (ap_int<9>)a0;
    ap_int<27> d = (ap_int<27>)a1;
    ap_int<18> b = (ap_int<18>)b0;

    ap_int<44> prod;
    prod = (a+d)*b;

    p0 += prod(35, 18) + prod(17,17);
    p1 += prod(17, 0);
}

void mac2dsp_int4(
    ap_int<4> a0, ap_int<4> a1,
    ap_int<4> b0, ap_int<4> b1,
    ap_int<8> &a0b0, ap_int<8> &a0b1,
    ap_int<8> &a1b0, ap_int<8> &a1b1
){

    a0b0 = a0 * b0;
    a0b1 = a0 * b1;
    a1b0 = a1 * b0;
    a1b1 = a1 * b1;

    // Shared DSP implementation
    // ap_int<27> a = (ap_int<27>)a0;
    // ap_int<27> d = 0;
    // d(26, 22) = (ap_int<5>)a1;

    // ap_int<18> b = 0;
    // b(3,0) = b0;
    // b(14, 11) = b1;

    // ap_int<44> prod;
    // prod = (a+d)*b;


    // a0b0 = prod(7, 0);
    // a0b1 = prod(18, 11);
    // a1b0 = prod(29, 22);
    // a1b1 = prod(40, 33);
}


