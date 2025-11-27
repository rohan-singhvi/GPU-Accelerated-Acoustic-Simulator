#include "simulation.h"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <atomic>

// --- HELPER: Ray-Triangle Intersection (Möller–Trumbore) ---
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

    float3 h;
    h.x = ray_dir.y * e2.z - ray_dir.z * e2.y;
    h.y = ray_dir.z * e2.x - ray_dir.x * e2.z;
    h.z = ray_dir.x * e2.y - ray_dir.y * e2.x;

    float a = dot(e1, h);
    if (a > -epsilon && a < epsilon) return 1e20f; 

    float f = 1.0f / a;
    float3 s = ray_origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return 1e20f;

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
    const float LISTENER_RADIUS = 0.5f; 

    std::cout << "Running CPU Simulation (OpenMP Threads: " << omp_get_max_threads() << ")..." << std::endl;
    std::cout << "Materials: Abs=" << params.wall_absorption 
              << ", Trans=" << params.wall_transmission << std::endl;

    // Thread-local accumulation buffers
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<float>> thread_irs(num_threads, std::vector<float>(ir_len, 0.0f));
    std::atomic<int> total_hits{0};

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();
        
        unsigned int seed = i * 747796405 + 2891336453;
        auto rand_float = [&seed]() { 
            seed = seed * 1103515245 + 12345;
            return (float)((seed / 65536) % 32768) / 32768.0f; 
        };

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
             int hit_tri_idx = -1;

             // --- GEOMETRY INTERSECTION LOGIC ---
             if (params.room_type == SHOEBOX) {
                 // (Shoebox logic same as before, omitted for brevity but should be kept if re-pasting entire file)
                 // Re-using logic from previous implementation
                if (dx.x > 0.0f) { float d = (params.room_dims.x - px.x) / dx.x; if (d < min_dist) { min_dist = d; nx = make_float3(-1, 0, 0); } } 
                else { float d = (0.0f - px.x) / dx.x; if (d < min_dist) { min_dist = d; nx = make_float3(1, 0, 0); } }
                if (dx.y > 0.0f) { float d = (params.room_dims.y - px.y) / dx.y; if (d < min_dist) { min_dist = d; nx = make_float3(0, -1, 0); } }
                else { float d = (0.0f - px.y) / dx.y; if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); } }
                if (dx.z > 0.0f) { float d = (params.room_dims.z - px.z) / dx.z; if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, -1); } }
                else { float d = (0.0f - px.z) / dx.z; if (d < min_dist) { min_dist = d; nx = make_float3(0, 0, 1); } }
             }
             else if (params.room_type == DOME) {
                float radius = params.room_dims.x;
                if (dx.y < 0.0f) { float d = (0.0f - px.y) / dx.y; if (d < min_dist) { min_dist = d; nx = make_float3(0, 1, 0); } }
                float b = 2.0f * dot(px, dx); float c = dot(px, px) - radius * radius; float disc = b*b - 4.0f*c;
                if(disc >= 0.0f) {
                    float sqrt_disc = sqrtf(disc);
                    float t = (-b - sqrt_disc)/2.0f;
                    if(t > 1e-3f && t < min_dist) { min_dist = t; nx = normalize(px + dx * t); }
                }
             }
             else if (params.room_type == MESH) {
                 float3 temp_n;
                 for(int t=0; t<mesh.num_triangles; ++t) {
                     float dist = intersect_triangle_cpu(px, dx, mesh.v0[t], mesh.v1[t], mesh.v2[t], mesh.normals[t], temp_n);
                     if (dist < min_dist) { min_dist = dist; nx = temp_n; hit_tri_idx = t; }
                 }
             }

             // --- LISTENER HIT CHECK ---
             float3 to_l = params.listener_pos - px;
             float t_proj = dot(to_l, dx);
             if (t_proj > 0 && t_proj < min_dist) {
                 float3 closest = px + dx * t_proj;
                 if (length_sq(params.listener_pos - closest) < LISTENER_RADIUS*LISTENER_RADIUS) {
                     float total_dist = dist_traveled + t_proj;
                     int idx = (int)((total_dist / SPEED_OF_SOUND) * SAMPLE_RATE);
                     if (idx < ir_len) {
                         thread_irs[tid][idx] += energy;
                         total_hits++;
                     }
                 }
             }

             if (min_dist >= 1e19f) break;

             // --- PHYSICS ENGINE: MATERIAL INTERACTION ---
             float3 hit_point = px + dx * min_dist;
             dist_traveled += min_dist;

             // Deciding Fate: Transmit or Reflect?
             float roll = rand_float(); // 0.0 to 1.0

             if (roll < params.wall_transmission) {
                 // --- TRANSMISSION (Pass Through) ---
                 // Ray continues in same direction (dx unchanged)
                 // We push the origin slightly past the wall to avoid self-intersecting the same spot
                 px = hit_point + dx * 0.01f; 
                 
                 // Transmission Loss: Passing through usually loses more energy than reflecting
                 energy *= 0.7f; 
                 
                 // (Optional: You could refract here using Snell's law if you wanted strictly accurate physics)
             } 
             else {
                 // --- REFLECTION (Bounce) ---
                 float d_dot_n = dot(dx, nx);
                 float3 reflection = dx - 2.0f * d_dot_n * nx;
                 dx = normalize(reflection);
                 
                 // Push off surface
                 px = hit_point + dx * 0.001f;
                 
                 // Apply Reflection Absorption
                 energy *= (1.0f - params.wall_absorption);
             }

             if (energy < 0.001f) break;
        }
    }
    
    std::cout << "Simulation Complete. Total Listener Hits: " << total_hits << std::endl;
    for(int t=0; t<num_threads; ++t) {
        for(int i=0; i<ir_len; ++i) {
            ir[i] += thread_irs[t][i];
        }
    }
}