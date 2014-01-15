#pragma once

#include <cuda_runtime_api.h>
#include <vector_types.h>

static float3* device_pos = NULL;
static float3* device_pout = NULL;
static float3* device_vel = NULL;
static float3* device_vout = NULL;
static float* device_masses = NULL;

void integrateRK4(float3* positions, float3* velocities, const float* masses, const float& dt, const int& N);

inline void cleanupCuda() {
    cudaFree(device_pos);
    cudaFree(device_vel);
    cudaFree(device_pout);
    cudaFree(device_vout);
    cudaFree(device_masses);
}
