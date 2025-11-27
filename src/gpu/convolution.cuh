#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>

// Convolves signal with impulse_response using FFT
// Result is mixed: dry * (1-mix) + wet * mix
std::vector<float> apply_reverb_gpu(
    const std::vector<float>& dry_signal, 
    const std::vector<float>& impulse_response, 
    float mix
);

#endif