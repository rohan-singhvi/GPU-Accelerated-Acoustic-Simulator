import os
import sys
import time
import math
import argparse
import numpy as np
from numba import cuda

IS_SIMULATION = os.environ.get('NUMBA_ENABLE_CUDASIM') == '1'
if IS_SIMULATION:
    import numpy as xp
    print("RUNNING IN SIMULATION MODE (CPU)")
else:
    import cupy as xp
    print("RUNNING IN REAL CUDA MODE (GPU)")

# ==========================================
#   DEVICE FUNCTIONS (GPU MATH HELPERS)
# ==========================================

@cuda.jit(device=True)
def reflect_vector(dx, dy, dz, nx, ny, nz):
    """Reflects vector D off Surface Normal N."""
    dot = dx*nx + dy*ny + dz*nz
    rx = dx - 2.0 * dot * nx
    ry = dy - 2.0 * dot * ny
    rz = dz - 2.0 * dot * nz
    return rx, ry, rz

@cuda.jit(device=True)
def intersect_sphere(px, py, pz, dx, dy, dz, radius):
    """Ray-Sphere Intersection. Returns distance or 1e20 if miss."""
    b = 2.0 * (px*dx + py*dy + pz*dz)
    c = (px*px + py*py + pz*pz) - (radius*radius)
    discriminant = b*b - 4.0*c
    
    if discriminant < 0.0: return 1e20
        
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0
    
    if t1 > 1e-3: return t1
    if t2 > 1e-3: return t2
    return 1e20

@cuda.jit(device=True)
def check_listener_hit(ray_x, ray_y, ray_z, dx, dy, dz, move_dist, lx, ly, lz, l_radius):
    """Checks if ray passes through the listener sphere."""
    rx = lx - ray_x
    ry = ly - ray_y
    rz = lz - ray_z
    t_projection = rx*dx + ry*dy + rz*dz
    
    if t_projection < 0.0 or t_projection > move_dist:
        return -1.0 # Miss
        
    cx = ray_x + dx * t_projection
    cy = ray_y + dy * t_projection
    cz = ray_z + dz * t_projection
    
    dist_sq = (lx - cx)**2 + (ly - cy)**2 + (lz - cz)**2
    if dist_sq < (l_radius * l_radius):
        return t_projection
    return -1.0

# ==========================================
#   THE KERNEL (PHYSICS ENGINE)
# ==========================================

@cuda.jit
def ray_trace_kernel(rays_pos, rays_dir, hits_log, room_dims, listener_pos, impulse_response, room_type):    
    # CONSTANTS
    SPEED_OF_SOUND = 343.0
    SAMPLE_RATE = 44100.0
    LISTENER_RADIUS = 0.2 # 20cm microphone
    
    idx = cuda.grid(1)
    # dont use more threads than we need to
    if idx < rays_pos.shape[0]:
        # Load State
        px, py, pz = rays_pos[idx, 0], rays_pos[idx, 1], rays_pos[idx, 2]
        dx, dy, dz = rays_dir[idx, 0], rays_dir[idx, 1], rays_dir[idx, 2]
        lx, ly, lz = listener_pos[0], listener_pos[1], listener_pos[2]
        
        dist_traveled = 0.0
        energy = 1.0
        
        for bounce in range(50): # Max Bounces
            
            # Init Hit Data
            min_dist = 1e20
            nx, ny, nz = 0.0, 0.0, 0.0 # Surface Normal
            
            # ---------------------------------------
            #  LOGIC A: SHOEBOX (Room Type 0)
            # ---------------------------------------
            if room_type == 0:
                room_x, room_y, room_z = room_dims[0], room_dims[1], room_dims[2]
                
                # Check all 6 planes
                # We calculate distance AND Normal vector simultaneously
                
                # X-Walls
                if dx > 0.0: 
                    d = (room_x - px) / dx
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, -1.0, 0.0, 0.0
                elif dx < 0.0:
                    d = (0.0 - px) / dx
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 1.0, 0.0, 0.0
                        
                # Y-Walls
                if dy > 0.0: 
                    d = (room_y - py) / dy
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 0.0, -1.0, 0.0
                elif dy < 0.0:
                    d = (0.0 - py) / dy
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 0.0, 1.0, 0.0

                # Z-Walls
                if dz > 0.0: 
                    d = (room_z - pz) / dz
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 0.0, 0.0, -1.0
                elif dz < 0.0:
                    d = (0.0 - pz) / dz
                    if d < min_dist:
                        min_dist, nx, ny, nz = d, 0.0, 0.0, 1.0

            # ---------------------------------------
            #  LOGIC B: DOME / HEMISPHERE (Room Type 1)
            # ---------------------------------------
            elif room_type == 1:
                radius = room_dims[0]
                
                # Floor (Plane at Y=0)
                if dy < 0.0:
                    d_floor = (0.0 - py) / dy
                    if d_floor < min_dist:
                        min_dist = d_floor
                        nx, ny, nz = 0.0, 1.0, 0.0 # Points Up
                
                # Dome (Sphere)
                d_sphere = intersect_sphere(px, py, pz, dx, dy, dz, radius)
                if d_sphere < min_dist:
                    min_dist = d_sphere
                    # Calculate Normal for Sphere (Point - Center)
                    # Center is 0,0,0
                    hit_x = px + dx * min_dist
                    hit_y = py + dy * min_dist
                    hit_z = pz + dz * min_dist
                    nx = hit_x / radius
                    ny = hit_y / radius
                    nz = hit_z / radius

            # ---------------------------------------
            #  SHARED PHYSICS (Applied to all Rooms)
            # ---------------------------------------
            # safety check - we missed universe?
            if min_dist >= 1e19:
                break
            # Did we pass the listener?
            hit_dist = check_listener_hit(px, py, pz, dx, dy, dz, min_dist, lx, ly, lz, LISTENER_RADIUS)
            
            if hit_dist > 0.0:
                total_dist = dist_traveled + hit_dist
                idx_time = int((total_dist / SPEED_OF_SOUND) * SAMPLE_RATE)
                if idx_time < impulse_response.shape[0]:
                    cuda.atomic.add(impulse_response, idx_time, energy)

            # Move Ray
            move_dist = min_dist + 1e-3
            px += dx * move_dist
            py += dy * move_dist
            pz += dz * move_dist
            dist_traveled += min_dist
            
            # Reflect (Using the Normal we calculated above)
            dx, dy, dz = reflect_vector(dx, dy, dz, nx, ny, nz)
            
            # Absorb Energy
            energy *= 0.85 
            if energy < 0.001: break
            
            # Debug Print (Thread 0 only)
            if idx == 0 and bounce < 3:
                 print("T0 Bounce", bounce, "Dist:", min_dist, "Type:", room_type)

        hits_log[idx] = dist_traveled

# ==========================================
#   HOST CODE (SETUP & CLI)
# ==========================================

def parse_vec3(s):
    """Helper to parse '1,2,3' into numpy array"""
    try:
        return np.array([float(x) for x in s.split(',')], dtype=np.float32)
    except:
        print(f"Error parsing vector: {s}")
        sys.exit(1)

def generate_rays_host(num_rays, source_pos):
    print(f"Generatings {num_rays} rays...")
    positions = np.tile(source_pos, (num_rays, 1)).astype(np.float32)
    random_vectors = np.random.normal(size=(num_rays, 3)).astype(np.float32)
    lengths = np.sqrt(np.sum(random_vectors**2, axis=1, keepdims=True))
    directions = random_vectors / lengths
    return positions, directions

def main():
    parser = argparse.ArgumentParser(description="GPU Acoustic Ray Tracer")
    
    # CLI Arguments
    parser.add_argument("--room", type=str, choices=['shoebox', 'dome'], default='shoebox', help="Room Shape")
    parser.add_argument("--rays", type=int, default=100_000, help="Number of Rays")
    parser.add_argument("--dims", type=str, default="10,5,8", help="Room Dimensions (L,W,H) or Radius")
    parser.add_argument("--source", type=str, default="2.0,1.5,1.5", help="Source Pos x,y,z")
    parser.add_argument("--listener", type=str, default="8.0,1.5,1.5", help="Listener Pos x,y,z")
    parser.add_argument("--out", type=str, default="room_impulse.wav", help="Output filename")
    
    args = parser.parse_args()
    
    # Config
    NUM_RAYS = args.rays
    SOURCE_POS = parse_vec3(args.source)
    LISTENER_POS = parse_vec3(args.listener)
    
    # Setup Room Data
    ROOM_TYPE_ID = 0
    if args.room == 'shoebox':
        ROOM_TYPE_ID = 0
        ROOM_DIMS = parse_vec3(args.dims) # expects "10,5,3"
    elif args.room == 'dome':
        ROOM_TYPE_ID = 1
        # For dome, we just need radius (first val of dims)
        radius = float(args.dims.split(',')[0])
        ROOM_DIMS = np.array([radius, 0.0, 0.0], dtype=np.float32)
        print(f"Configuring Dome with Radius {radius}m")

    # Prep Data
    h_pos, h_dir = generate_rays_host(NUM_RAYS, SOURCE_POS)
    
    # Move to GPU
    d_pos = cuda.to_device(h_pos)
    d_dir = cuda.to_device(h_dir)
    d_hits = cuda.to_device(np.zeros(NUM_RAYS, dtype=np.float32))
    d_room_dims = cuda.to_device(ROOM_DIMS)
    d_listener_pos = cuda.to_device(LISTENER_POS)
    
    # Audio Buffer
    ir_len = 44100 # 1 second
    d_impulse_response = cuda.to_device(xp.zeros(ir_len, dtype=np.float32))

    # 3. Launch
    threads_per_block = 256
    blocks = (NUM_RAYS + (threads_per_block - 1)) // threads_per_block
    
    print(f"Launching Kernel: {blocks} blocks | {args.room.upper()}")
    start_time = time.time()
    
    ray_trace_kernel[blocks, threads_per_block](
        d_pos, d_dir, d_hits, d_room_dims, d_listener_pos, d_impulse_response, ROOM_TYPE_ID
    )
    
    if not IS_SIMULATION:
        cuda.synchronize()
        
    print(f"Finished in {time.time() - start_time:.4f}s")
    
    final_ir = d_impulse_response.copy_to_host()
    max_val = np.max(final_ir)
    print(f"Max Amplitude: {max_val}")
    
    if max_val > 0:
        final_ir = final_ir / max_val
        
    import scipy.io.wavfile
    wav_data = (final_ir * 32767).astype(np.int16)
    scipy.io.wavfile.write(args.out, 44100, wav_data)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()