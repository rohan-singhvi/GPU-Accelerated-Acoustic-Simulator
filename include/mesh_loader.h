#ifndef MESH_LOADER_H
#define MESH_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

// CRITICAL FIX: Do NOT include <cuda_runtime.h> directly.
// Include our wrapper "cuda_math.h" instead. 
// This ensures we get 'float3' even if CUDA is disabled.
#include "cuda_math.h"

struct MeshData {
    std::vector<float3> v0;
    std::vector<float3> v1;
    std::vector<float3> v2;
    std::vector<float3> normals;
    int num_triangles = 0;
};

// A simple, dependency-free OBJ loader
inline MeshData load_obj(const std::string& filename) {
    MeshData mesh;
    std::vector<float3> temp_vertices;
    // temp_normals usage removed or simplified for this specific loader approach

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open mesh file " << filename << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string prefix;
        ss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            temp_vertices.push_back(make_float3(x, y, z));
        } 
        else if (prefix == "f") {
            // Very basic face parsing "f v1 v2 v3" or "f v1//n1 v2//n2 v3//n3"
            std::string segment;
            int v_indices[3];
            int i = 0;
            while (ss >> segment && i < 3) {
                size_t slash = segment.find('/');
                std::string v_str = (slash != std::string::npos) ? segment.substr(0, slash) : segment;
                v_indices[i] = std::stoi(v_str) - 1; // OBJ is 1-indexed
                i++;
            }

            if (i == 3) {
                // Bounds check
                if(v_indices[0] < temp_vertices.size() && 
                   v_indices[1] < temp_vertices.size() && 
                   v_indices[2] < temp_vertices.size()) 
                {
                    float3 p0 = temp_vertices[v_indices[0]];
                    float3 p1 = temp_vertices[v_indices[1]];
                    float3 p2 = temp_vertices[v_indices[2]];

                    mesh.v0.push_back(p0);
                    mesh.v1.push_back(p1);
                    mesh.v2.push_back(p2);
                    mesh.num_triangles++;

                    // Calculate Face Normal
                    float3 e1 = p1 - p0;
                    float3 e2 = p2 - p0;
                    
                    float3 n;
                    n.x = e1.y * e2.z - e1.z * e2.y;
                    n.y = e1.z * e2.x - e1.x * e2.z;
                    n.z = e1.x * e2.y - e1.y * e2.x;
                    
                    float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
                    if (len > 1e-6f) {
                        n.x /= len; n.y /= len; n.z /= len;
                    }
                    mesh.normals.push_back(n);
                }
            }
        }
    }
    
    std::cout << "Loaded " << mesh.num_triangles << " triangles from " << filename << std::endl;
    return mesh;
}

#endif