#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>

std::vector<float> apply_reverb(const std::vector<float>& dry, const std::vector<float>& ir,
                                float mix);

std::vector<float> apply_reverb_cpu(const std::vector<float>& dry, const std::vector<float>& ir,
                                    float mix);

#ifdef ENABLE_CUDA
std::vector<float> apply_reverb_gpu(const std::vector<float>& dry, const std::vector<float>& ir,
                                    float mix);
#endif

#endif