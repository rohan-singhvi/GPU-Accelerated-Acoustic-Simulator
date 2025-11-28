#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include <cuda_runtime.h>

#include <vector>

#include "cuda_math.h"
#include "mesh_loader.h"

enum RoomType { SHOEBOX = 0, DOME = 1, MESH = 2 };

struct SimulationParams {
    int num_rays;
    RoomType room_type;
    float3 room_dims;  // Box: L,W,H | Dome: R,0,0
    float3 source_pos;
    float3 listener_pos;
    std::string mesh_path;
};

// Host wrapper to launch kernel
void run_acoustic_simulation(const SimulationParams& params, const MeshData& mesh,
                             std::vector<float>& h_impulse_response);

#endif