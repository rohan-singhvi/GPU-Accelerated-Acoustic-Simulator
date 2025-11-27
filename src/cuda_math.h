#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <cmath>

// If CUDA is available, use its built-in types
#if defined(__CUDACC__) || defined(ENABLE_CUDA)
    #include <cuda_runtime.h>
    #include <vector_types.h>
    
    // Check if we are in a pure C++ file (not NVCC) but have CUDA enabled
    // cuda_runtime.h gives us float3, but we might need to define operators manually for host C++ code
    // if vector_functions.h isn't included or doesn't provide host operators.
#else
    // CPU-Only Fallback: Define float3 manually
    struct float3 {
        float x, y, z;
    };
    
    inline float3 make_float3(float x, float y, float z) {
        return {x, y, z};
    }
#endif

//  Shared Vector Math Operators (Host & Device) 
// We define these as inline functions that work on both CPU and GPU

#if defined(__CUDACC__)
    #define HD __host__ __device__
#else
    #define HD inline
#endif

HD float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

HD float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

HD float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

HD float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

HD float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

HD float length(const float3& a) {
    return sqrtf(dot(a, a));
}

HD float length_sq(const float3& a) {
    return dot(a, a);
}

HD float3 normalize(const float3& a) {
    float len = length(a);
    if (len < 1e-6f) return make_float3(0.0f, 0.0f, 0.0f);
    return a * (1.0f / len);
}

#endif