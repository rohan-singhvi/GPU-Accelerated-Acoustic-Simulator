#include "simulation.h"
#include <cmath>
#include <omp.h> // for multithreading
#include <iostream>
#include <algorithm>
#include <atomic> // Added for hit counting

// Ray-Triangle Intersection (Möller–Trumbore) 
// Returns distance t, or 1e20 if miss. Updates normal if hit.
inline float intersect_triangle_cpu(
    const float3& ray_origin, const float3& ray_dir,
    const float3& v0, const float3& v1, const float3& v2,
    const float3& tri_normal,
    float3& out_normal
) {
    const float epsilon = 1e-6f;
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    // h = ray_dir X e2
    float3 h;
    h.x = ray_dir.y * e2.z - ray_dir.z * e2.y;
    h.y = ray_dir.z * e2.x - ray_dir.x * e2.z;
    h.z = ray_dir.x * e2.y - ray_dir.y * e2.x;

    float a = dot(e1, h);
    // intentionally allow negative 'a' (backfaces) to ensure we hit 
    // the mesh even if we are inside it or normals are inverted.
    if (a > -epsilon && a < epsilon) return 1e20f; // Parallel

    float f = 1.0f / a;
    float3 s = ray_origin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return 1e20f;

    // q = s X e1
    float3 q;
    q.x = s.y * e1.z - s.z * e1.y;
    q.y = s.z * e1.x - s.x * e1.z;
    q.z = s.x * e1.y - s.y * e1.x;

    float v = f * dot(ray_dir, q);
    if (v < 0.0f || u + v > 1.0f) return 1e20f;

    float t = f * dot(e2, q);

    if (t > 1e-3f) {
        out_normal = tri_normal;
        return t;
    }
    return 1e20f;
}

void run_simulation_cpu(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir) {
    int N = params.num_rays;
    int ir_len = 44100; // 1 second
    ir.resize(ir_len, 0.0f);

    const float SPEED_OF_SOUND = 343.0f;
    const float SAMPLE_RATE = 44100.0f;
    // 0.5 to ensure hits with lower ray counts
    const float LISTENER_RADIUS = 0.5f; 

    std::cout << "Running CPU Simulation (OpenMP Threads: " << omp_get_max_threads() << ")..." << std::endl;
    if (params.room_type == MESH) {
        std::cout << "Mesh Mode: Checking " << mesh.num_triangles << " triangles per ray." << std::endl;
    }

    // Thread-local accumulation buffers
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<float>> thread_irs(num_threads, std::vector<float>(ir_len, 0.0f));
    
    // Global Hit Counter for Debugging
    std::atomic<int> total_hits{0};

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();
        
        // Simple Pseudo-Random Number Generator (PRNG) per ray
        unsigned int seed = i * 747796405 + 2891336453;
        auto rand_float = [&seed]() { 
            seed = seed * 1103515245 + 12345;
            return (float)((seed / 65536) % 32768) / 32768.0f; 
        };

        // random direction vector
        float u = rand_float() * 2.0f - 1.0f;
        float v = rand_float() * 2.0f - 1.0f;
        float w = rand_float() * 2.0f - 1.0f;
        float3 dx = normalize(make_float3(u, v, w));
        float3 px = params.source_pos;

        float dist_traveled = 0.0f;
        float energy = 1.0f;

        for (int bounce = 0; bounce < 50; ++bounce) {
             float min_dist = 1e20f;
             float3 nx = make_float3(0,0,0);
             int hit_tri_idx = -1; // debug

             //  shoebox (put this in a function)
             if (params.room_type == SHOEBOX) {
                if (dx.x > 0.0f) {
                    float d = (params.room_dims.x - px.x) / dx.x;
                    if (d < min_dist) { min_dist = d; nx = make_float3(-1, 0, 0); }
                } else {
                    float d = (0.0f - px.x) / dx.x;
                    if (d < min_dist) { min_dist = d; nx = make_float3(1, 0, 0); }
                }
                if (dx.y > 0.0f) {
                    float d = (params.room_dims.y - px.y) / dx.y;
                    if (d < min_dist) { min_dist = d; nx = make_float3(0, -1, 0); }
                } else {
                    float d = (0.0f - px.y) / dx.y;
                    if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); }
                }
                if (dx.z > 0.0f) {
                    float d = (params.room_dims.z - px.z) / dx.z;
                    if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, -1); }
                } else {
                    float d = (0.0f - px.z) / dx.z;
                    if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, 1); }
                }
             }
             //  dome (put this in a function)
             else if (params.room_type == DOME) {
                float radius = params.room_dims.x;
                // Floor
                if (dx.y < 0.0f) {
                    float d = (0.0f - px.y) / dx.y;
                    if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); }
                }
                // Sphere Intersect 
                float b = 2.0f * dot(px, dx);
                float c = dot(px, px) - radius * radius;
                float disc = b*b - 4.0f*c;
                if(disc >= 0.0f) {
                    float sqrt_disc = sqrtf(disc);
                    float t1 = (-b - sqrt_disc)/2.0f;
                    float t2 = (-b + sqrt_disc)/2.0f;
                    float t = (t1 > 1e-3f) ? t1 : ((t2 > 1e-3f) ? t2 : 1e20f);
                    if(t < min_dist) {
                        min_dist = t;
                        nx = normalize(px + dx * t);
                    }
                }
             }
             //  mesh (put this in a function) 
             else if (params.room_type == MESH) {
                 float3 temp_n;
                 for(int t=0; t<mesh.num_triangles; ++t) {
                     float dist = intersect_triangle_cpu(
                         px, dx, 
                         mesh.v0[t], mesh.v1[t], mesh.v2[t], 
                         mesh.normals[t], 
                         temp_n
                     );
                     
                     if (dist < min_dist) {
                         min_dist = dist;
                         nx = temp_n;
                         hit_tri_idx = t;
                     }
                 }
             }

             //  listener hit?
             float3 to_l = params.listener_pos - px;
             float t_proj = dot(to_l, dx);
             
             // if listener is in front of us and closer than the wall
             if (t_proj > 0 && t_proj < min_dist) {
                 float3 closest = px + dx * t_proj;
                 float dist_sq = length_sq(params.listener_pos - closest);
                 if (dist_sq < LISTENER_RADIUS*LISTENER_RADIUS) {
                     float total_dist = dist_traveled + t_proj;
                     int idx = (int)((total_dist / SPEED_OF_SOUND) * SAMPLE_RATE);
                     if (idx < ir_len) {
                         thread_irs[tid][idx] += energy;
                         total_hits++; // Count the hit!
                     }
                 }
             }
             if (min_dist >= 1e19f) {
                 if (i == 0 && bounce == 0) printf("[Ray 0] Missed Mesh completely.\n");
                 break;
             }

             // debug ray 0
             if (i == 0 && bounce < 3) {
                 printf("[Ray 0] Bounce %d: Hit Wall at %.2f (Tri: %d)\n", bounce, min_dist, hit_tri_idx);
             }

             // move along reflection
             float3 hit_point = px + dx * min_dist;
             dist_traveled += min_dist;

             float d_dot_n = dot(dx, nx);
             float3 reflection = dx - 2.0f * d_dot_n * nx;
             
             // update dir (normalize to prevent float error accumulation)
             dx = normalize(reflection);

             // nudge
             px = hit_point + dx * 0.001f;

             // absorb 15%
             energy *= 0.85f;
             if (energy < 0.001f) break;
        }
    }

    std::cout << "Simulation Complete. Total Listener Hits: " << total_hits << std::endl;
    if (total_hits == 0) {
        std::cout << "WARNING: No rays hit the listener! Try increasing --rays or the listener size is too small." << std::endl;
    }

    for(int t=0; t<num_threads; ++t) {
        for(int i=0; i<ir_len; ++i) {
            ir[i] += thread_irs[t][i];
        }
    }
}