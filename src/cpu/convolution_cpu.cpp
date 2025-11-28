#include <algorithm>
#include <iostream>

#include "convolution.h"

#ifdef USE_FFTW
#include <fftw3.h>
#endif

std::vector<float> apply_reverb_cpu(const std::vector<float>& dry, const std::vector<float>& ir,
                                    float mix) {
    int n_dry = dry.size();
    int n_ir = ir.size();
    int n_out = n_dry + n_ir - 1;
    std::vector<float> result(n_out);

#ifdef USE_FFTW
    std::cout << "Using FFTW3 for CPU Convolution..." << std::endl;
    // Next power of 2
    int fft_size = 1;
    while (fft_size < n_out) fft_size *= 2;

    // Allocate FFTW buffers
    float* in_dry = fftwf_alloc_real(fft_size);
    float* in_ir = fftwf_alloc_real(fft_size);
    fftwf_complex* out_dry = fftwf_alloc_complex(fft_size / 2 + 1);
    fftwf_complex* out_ir = fftwf_alloc_complex(fft_size / 2 + 1);
    float* out_final = fftwf_alloc_real(fft_size);

    // Plans
    fftwf_plan p_fwd_dry = fftwf_plan_dft_r2c_1d(fft_size, in_dry, out_dry, FFTW_ESTIMATE);
    fftwf_plan p_fwd_ir = fftwf_plan_dft_r2c_1d(fft_size, in_ir, out_ir, FFTW_ESTIMATE);
    fftwf_plan p_inv = fftwf_plan_dft_c2r_1d(fft_size, out_dry, out_final, FFTW_ESTIMATE);

    // Prepare Data
    std::fill(in_dry, in_dry + fft_size, 0.0f);
    std::fill(in_ir, in_ir + fft_size, 0.0f);
    std::copy(dry.begin(), dry.end(), in_dry);
    std::copy(ir.begin(), ir.end(), in_ir);

    // Execute Forward
    fftwf_execute(p_fwd_dry);
    fftwf_execute(p_fwd_ir);

    // Complex Multiply
    int complex_size = fft_size / 2 + 1;
    float scale = 1.0f / (float)fft_size;

    for (int i = 0; i < complex_size; ++i) {
        float ar = out_dry[i][0];
        float ai = out_dry[i][1];
        float br = out_ir[i][0];
        float bi = out_ir[i][1];

        // (a+bi)(c+di)
        out_dry[i][0] = (ar * br - ai * bi) * scale;
        out_dry[i][1] = (ar * bi + ai * br) * scale;
    }

    // Execute Inverse
    fftwf_execute(p_inv);

    // Mix
    for (int i = 0; i < n_out; ++i) {
        float d = (i < n_dry) ? dry[i] : 0.0f;
        result[i] = d * (1.0f - mix) + out_final[i] * mix;
    }

    fftwf_destroy_plan(p_fwd_dry);
    fftwf_destroy_plan(p_fwd_ir);
    fftwf_destroy_plan(p_inv);
    fftwf_free(in_dry);
    fftwf_free(in_ir);
    fftwf_free(out_dry);
    fftwf_free(out_ir);
    fftwf_free(out_final);

#else
    std::cout << "WARNING: No FFTW3 found. Using SLOW naive convolution." << std::endl;
    // Super naive fallback
    for (int i = 0; i < n_dry; i++) {
        for (int j = 0; j < n_ir; j++) {
            result[i + j] += dry[i] * ir[j];
        }
    }
    // Simple mix logic
#endif

    return result;
}