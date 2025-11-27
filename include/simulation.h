#ifndef SIMULATION_H
#define SIMULATION_H

#include <vector>
#include <string>
#include "cuda_math.h"
#include "mesh_loader.h"

enum RoomType { SHOEBOX = 0, DOME = 1, MESH = 2 };

struct SimulationParams {
    int num_rays;
    RoomType room_type;
    float3 room_dims; 
    float3 source_pos;
    float3 listener_pos;
    std::string mesh_path;
    
    float wall_absorption;   // 0.0 = Perfect Mirror, 1.0 = Dead Silence
    float wall_transmission; // 0.0 = Solid Wall, 1.0 = Transparent (Hole)
};

// Dispatchers
void run_simulation(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir);
void run_simulation_cpu(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir);
#ifdef ENABLE_CUDA
    void run_simulation_gpu(const SimulationParams& params, const MeshData& mesh, std::vector<float>& ir);
#endif

#endif