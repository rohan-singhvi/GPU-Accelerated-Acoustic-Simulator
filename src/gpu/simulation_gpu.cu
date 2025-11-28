#include "simulation.cuh"
#include "cuda_math.h"
#include <cstdio>

#define SPEED_OF_SOUND 343.0f
#define SAMPLE_RATE 44100.0f
#define LISTENER_RADIUS 0.2f
#define MAX_BOUNCES 50

__device__ void intersect_sphere(
    const float3& ray_origin, const float3& ray_dir, 
    float radius, 
    float& min_dist, float3& normal
) {
    // Ray-Sphere intersection
    float b = 2.0f * dot(ray_origin, ray_dir);
    float c = dot(ray_origin, ray_origin) - (radius * radius);
    float disc = b * b - 4.0f * c;

    if (disc < 0.0f) return;

    float sqrt_disc = sqrtf(disc);
    float t1 = (-b - sqrt_disc) / 2.0f;
    float t2 = (-b + sqrt_disc) / 2.0f;

    float t = 1e20f;
    if (t1 > 1e-3f) t = t1;
    else if (t2 > 1e-3f) t = t2;

    if (t < min_dist) {
        min_dist = t;
        float3 hit_point = ray_origin + ray_dir * t;
        normal = normalize(hit_point); // Sphere at 0,0,0 normal is just normalized pos
    }
}

__device__ void intersect_triangle(
    const float3& ray_origin, const float3& ray_dir,
    const float3& v0, const float3& v1, const float3& v2,
    const float3& tri_normal,
    float& min_dist, float3& normal
) {
    const float epsilon = 1e-6f;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    float3 h;
    h.x = ray_dir.y * e2.z - ray_dir.z * e2.y;
    h.y = ray_dir.z * e2.x - ray_dir.x * e2.z;
    h.z = ray_dir.x * e2.y - ray_dir.y * e2.x;

    float a = dot(e1, h);
    if (a > -epsilon && a < epsilon) return;

    float f = 1.0f / a;
    float3 s = ray_origin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return;

    float3 q;
    q.x = s.y * e1.z - s.z * e1.y;
    q.y = s.z * e1.x - s.x * e1.z;
    q.z = s.x * e1.y - s.y * e1.x;

    float v = f * dot(ray_dir, q);
    if (v < 0.0f || u + v > 1.0f) return;

    float t = f * dot(e2, q);

    if (t > 1e-3f && t < min_dist) {
        min_dist = t;
        normal = tri_normal;
    }
}

__global__ void ray_trace_kernel(
    float3* d_pos,
    float3* d_dir,
    float3 room_dims,
    float3 listener_pos,
    float* d_impulse_response,
    int room_type,
    int ir_length,
    // mesh data
    float3* d_v0, float3* d_v1, float3* d_v2, float3* d_normals, int num_triangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // bound check (assuming d_pos size matches num_rays)
    // can't easily check array size in CUDA, rely on launch bounds
    
    float3 px = d_pos[idx];
    float3 dx = d_dir[idx];
    
    float dist_traveled = 0.0f;
    float energy = 1.0f;

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce) {
        float min_dist = 1e20f;
        float3 nx = make_float3(0.0f, 0.0f, 0.0f);

        //  shoebox
        if (room_type == SHOEBOX) {
            // X-Walls
            if (dx.x > 0.0f) {
                float d = (room_dims.x - px.x) / dx.x;
                if (d < min_dist) { min_dist = d; nx = make_float3(-1, 0, 0); }
            } else {
                float d = (0.0f - px.x) / dx.x;
                if (d < min_dist) { min_dist = d; nx = make_float3(1, 0, 0); }
            }
            // Y-Walls
            if (dx.y > 0.0f) {
                float d = (room_dims.y - px.y) / dx.y;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, -1, 0); }
            } else {
                float d = (0.0f - px.y) / dx.y;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); }
            }
            // Z-Walls
            if (dx.z > 0.0f) {
                float d = (room_dims.z - px.z) / dx.z;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, -1); }
            } else {
                float d = (0.0f - px.z) / dx.z;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, 1); }
            }
        }
        //  dome 
        else if (room_type == DOME) {
            float radius = room_dims.x;
            // Floor
            if (dx.y < 0.0f) {
                float d = (0.0f - px.y) / dx.y;
                if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); }
            }
            // Dome
            intersect_sphere(px, dx, radius, min_dist, nx);
        }
        //  mesh 
        else if (room_type == MESH) {
            for(int i = 0; i < num_triangles; ++i) {
                float3 v0 = d_v0[i];
                float3 v1 = d_v1[i];
                float3 v2 = d_v2[i];

                // Calculate Geometric Normal on the fly
                // This ensures we always have a valid normal perpendicular to the face
                float3 e1 = v1 - v0;
                float3 e2 = v2 - v0;
                
                // Manual Cross Product (e1 x e2)
                float3 calculated_normal;
                calculated_normal.x = e1.y * e2.z - e1.z * e2.y;
                calculated_normal.y = e1.z * e2.x - e1.x * e2.z;
                calculated_normal.z = e1.x * e2.y - e1.y * e2.x;
                
                // Normalize it
                float len = sqrtf(dot(calculated_normal, calculated_normal));
                if (len > 1e-6f) {
                    float invLen = 1.0f / len;
                    calculated_normal.x *= invLen;
                    calculated_normal.y *= invLen;
                    calculated_normal.z *= invLen;
                }

                // Pass the calculated normals
                intersect_triangle(px, dx, v0, v1, v2, calculated_normal, min_dist, nx);
            }
        }
        
        // Missed everything?
        if (min_dist >= 1e19f) break;

        // listener intersection along this ray segment
        float3 to_listener = listener_pos - px;
        float t_proj = dot(to_listener, dx);

        if (t_proj > 0.0f && t_proj < min_dist) {
            float3 closest_point = px + dx * t_proj;
            float dist_sq = length_sq(listener_pos - closest_point);
            
            if (dist_sq < (LISTENER_RADIUS * LISTENER_RADIUS)) {
                // hit!
                float total_dist = dist_traveled + t_proj;
                int idx_time = (int)((total_dist / SPEED_OF_SOUND) * SAMPLE_RATE);
                
                if (idx_time < ir_length) {
                    atomicAdd(&d_impulse_response[idx_time], energy);
                }
            }
        }

        // move ray
        float move_dist = min_dist + 1e-3f; // nudge
        px = px + dx * move_dist;
        dist_traveled += min_dist;

        // reflect
        float dot_prod = dot(dx, nx);
        dx = dx - 2.0f * dot_prod * nx;

        // absorb
        energy *= 0.85f;
        if (energy < 0.001f) break;
    }
}

// wrapper
void run_simulation_gpu(const SimulationParams& params, const MeshData& mesh, std::vector<float>& h_impulse_response) {
    int N = params.num_rays;
    
    // generate rays
    std::vector<float3> h_pos(N);
    std::vector<float3> h_dir(N);
    
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        h_pos[i] = params.source_pos;
        
        // random spherical direction
        float u = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float w = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        float3 d = make_float3(u, v, w);
        h_dir[i] = normalize(d);
    }

    // allocate device memory
    float3 *d_pos, *d_dir, *d_v0 = nullptr, *d_v1 = nullptr, *d_v2 = nullptr, *d_normals = nullptr;
    float *d_ir;
    
    cudaMalloc(&d_pos, N * sizeof(float3));
    cudaMalloc(&d_dir, N * sizeof(float3));
    
    cudaMemcpy(d_pos, h_pos.data(), N * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dir, h_dir.data(), N * sizeof(float3), cudaMemcpyHostToDevice);

    // IR buffer
    int ir_len = 44100; // 1 second buffer
    cudaMalloc(&d_ir, ir_len * sizeof(float));
    cudaMemset(d_ir, 0, ir_len * sizeof(float));

    // mesh buffer (if needed)
    if (params.room_type == MESH) {
        int t_count = mesh.num_triangles;
        cudaMalloc(&d_v0, t_count * sizeof(float3));
        cudaMalloc(&d_v1, t_count * sizeof(float3));
        cudaMalloc(&d_v2, t_count * sizeof(float3));
        cudaMalloc(&d_normals, t_count * sizeof(float3));
        
        cudaMemcpy(d_v0, mesh.v0.data(), t_count * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v1, mesh.v1.data(), t_count * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, mesh.v2.data(), t_count * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_normals, mesh.normals.data(), t_count * sizeof(float3), cudaMemcpyHostToDevice);
    }
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    printf("Launching Kernel: %d rays, %d blocks\n", N, blocks);
    
    ray_trace_kernel<<<blocks, threads>>>(
        d_pos, d_dir, 
        params.room_dims, 
        params.listener_pos, 
        d_ir, 
        (int)params.room_type, 
        ir_len,
        d_v0, d_v1, d_v2, d_normals, mesh.num_triangles
    );
    
    cudaDeviceSynchronize();
    
    // errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    h_impulse_response.resize(ir_len);
    cudaMemcpy(h_impulse_response.data(), d_ir, ir_len * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_pos);
    cudaFree(d_dir);
    cudaFree(d_ir);
    if(d_v0) cudaFree(d_v0);
    if(d_v1) cudaFree(d_v1);
    if(d_v2) cudaFree(d_v2);
    if(d_normals) cudaFree(d_normals);
}
