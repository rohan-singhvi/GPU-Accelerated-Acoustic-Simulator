#include "convolution.cuh"
#include <algorithm>
#include <iostream>
#include <cmath>

// Kernel to multiply complex numbers: A *= B
// cuFFT stores complex numbers as float2 (x=real, y=imag)
__global__ void complex_multiply_kernel(cufftComplex* a, cufftComplex* b, int size, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float2 valA = a[i];
        float2 valB = b[i];
        
        // (a+bi)(c+di) = (ac - bd) + (ad + bc)i
        float real = valA.x * valB.x - valA.y * valB.y;
        float imag = valA.x * valB.y + valA.y * valB.x;
        
        a[i] = make_cuComplex(real * scale, imag * scale);
    }
}

// Kernel to mix wet and dry signals
__global__ void mix_kernel(float* out, float* wet, float* dry, int dry_len, float mix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // We process up to wet_len (which is > dry_len)
    
    // Safety check in caller ensures bounds, but we check here too
    float w_val = wet[i];
    float d_val = (i < dry_len) ? dry[i] : 0.0f;
    
    out[i] = (d_val * (1.0f - mix)) + (w_val * mix);
}

std::vector<float> apply_reverb_gpu(
    const std::vector<float>& dry_signal, 
    const std::vector<float>& impulse_response, 
    float mix
) {
    int n_dry = dry_signal.size();
    int n_ir = impulse_response.size();
    int n_out = n_dry + n_ir - 1;
    
    // Find next power of 2 for FFT efficiency
    int fft_size = 1;
    while (fft_size < n_out) fft_size *= 2;
    
    std::cout << "FFT Size: " << fft_size << " (Dry: " << n_dry << ", IR: " << n_ir << ")" << std::endl;

    // Allocate GPU Memory
    float *d_dry_padded, *d_ir_padded;
    cufftComplex *d_dry_freq, *d_ir_freq;
    
    cudaMalloc(&d_dry_padded, fft_size * sizeof(float));
    cudaMalloc(&d_ir_padded, fft_size * sizeof(float));
    
    // Clear memory (padding with zeros)
    cudaMemset(d_dry_padded, 0, fft_size * sizeof(float));
    cudaMemset(d_ir_padded, 0, fft_size * sizeof(float));
    
    // Copy data
    cudaMemcpy(d_dry_padded, dry_signal.data(), n_dry * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ir_padded, impulse_response.data(), n_ir * sizeof(float), cudaMemcpyHostToDevice);
    
    // FFT Plans
    // R2C: Real input -> Complex output (Size is fft_size/2 + 1)
    cufftHandle plan_fwd, plan_inv;
    cufftPlan1d(&plan_fwd, fft_size, CUFFT_R2C, 1);
    cufftPlan1d(&plan_inv, fft_size, CUFFT_C2R, 1);
    
    int complex_size = fft_size / 2 + 1;
    cudaMalloc(&d_dry_freq, complex_size * sizeof(cufftComplex));
    cudaMalloc(&d_ir_freq, complex_size * sizeof(cufftComplex));
    
    // 1. Forward FFT
    cufftExecR2C(plan_fwd, d_dry_padded, d_dry_freq);
    cufftExecR2C(plan_fwd, d_ir_padded, d_ir_freq);
    
    // 2. Complex Multiply (Convolution in Freq Domain)
    int threads = 256;
    int blocks = (complex_size + threads - 1) / threads;
    // Scale factor for IFFT (1/N)
    float scale = 1.0f / (float)fft_size;
    
    complex_multiply_kernel<<<blocks, threads>>>(d_dry_freq, d_ir_freq, complex_size, scale);
    
    // 3. Inverse FFT (Result goes back into d_dry_padded to save memory)
    // Note: C2R destroys input, which is fine here
    cufftExecC2R(plan_inv, d_dry_freq, d_dry_padded);
    
    cudaDeviceSynchronize();
    
    // 4. Mix Helper
    // We need a separate buffer for the final mix to ensure clean "Dry" data
    // Actually, d_dry_padded now holds the "Wet" signal.
    // We need the original dry signal again on GPU for the mix.
    float* d_dry_original;
    cudaMalloc(&d_dry_original, n_dry * sizeof(float));
    cudaMemcpy(d_dry_original, dry_signal.data(), n_dry * sizeof(float), cudaMemcpyHostToDevice);
    
    float* d_final;
    cudaMalloc(&d_final, n_out * sizeof(float));
    
    blocks = (n_out + threads - 1) / threads;
    mix_kernel<<<blocks, threads>>>(d_final, d_dry_padded, d_dry_original, n_dry, mix);
    
    // Copy back
    std::vector<float> result(n_out);
    cudaMemcpy(result.data(), d_final, n_out * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    cudaFree(d_dry_padded);
    cudaFree(d_ir_padded);
    cudaFree(d_dry_freq);
    cudaFree(d_ir_freq);
    cudaFree(d_dry_original);
    cudaFree(d_final);
    
    return result;
}