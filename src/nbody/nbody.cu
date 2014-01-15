#include <sys/time.h>
#include <iostream>
#include <cmath>
using namespace std;

#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "math/util.hpp"
#include "math/constants.hpp"

#include "nbody.hpp"

/**
  Compute force vector due to gravity from all other bodies on the given point.
*/
__device__ float3 gravity(
        const float3& point, const float3* bodies, const float* masses,
        const int& N) {

    float3 grav = make_float3(0,0,0);

    for(int j=0; j<N; ++j) {
        const float3 r = bodies[j] - point;

        // gravity force = G * m1*m2 / r^2
        // acceleration = F/m
        // a = G*m1*m2 / m1*r^2 = G*m2 / r^2
        if(norm2(r) > metersPerAU*metersPerAU) {
            const float acceleration = G * masses[j] / norm2(r);
            // The acceleration is a scalar, but it acts in the direction
            // pointing towards the other body.
            grav += acceleration * normalize(r);
        }
    }

    return grav;
}

// Controls the maximum amount of shared bodies (shared memory size in the
// kernel).
__device__ const int maxSharedBodies = 512;
__device__ const int threadsPerBlock = 128;

__global__ void rk4_kernel(
        const float3* pos, const float3* vel, const float* masses,
        float3* pos_out, float3* vel_out,
        const float dt, const int N){

    const int numBlocks = ceil((float)N / maxSharedBodies);
    const int blockOffset = blockIdx.x * maxSharedBodies;
    const int nextBlock = min((blockIdx.x + 1) * maxSharedBodies, N);
    const int bodiesPerThread = (nextBlock-blockOffset) / threadsPerBlock;

    memcpy(pos_out, pos, N*sizeof(float3));
    memcpy(vel_out, vel, N*sizeof(float3));

    __shared__ float3 sh_pos[maxSharedBodies];
    __shared__ float sh_mass[maxSharedBodies];

    // The shared memory is most likely smaller than the number of bodies, so
    // they are copied into shared memory in blocks that rotate over the body
    // list.
    for(int curBlock=0; curBlock < numBlocks; ++curBlock) {
        // Each thread copies the bodies that it's responsible for into shared
        // memory.
        for(int i = (curBlock*maxSharedBodies) + threadIdx.x*bodiesPerThread; i < (curBlock*maxSharedBodies) + (threadIdx.x+1)*bodiesPerThread; ++i) {
            sh_pos[i-(curBlock*maxSharedBodies)] = pos[i];
            sh_mass[i-(curBlock*maxSharedBodies)] = masses[i];
        }
        __syncthreads();

        // RK4 Updates.
        for(int i = threadIdx.x*bodiesPerThread; i < (threadIdx.x+1)*bodiesPerThread; ++i) {
            const float3 dxdt1 = vel_out[blockOffset+i];
            const float3 dvdt1 = gravity(pos_out[blockOffset+i], sh_pos, sh_mass, maxSharedBodies);

            const float3 dxdt2 = vel_out[blockOffset+i] + dt * dvdt1/2.f;
            const float3 dvdt2 = gravity(pos_out[blockOffset+i] + dt * dxdt1/2.f, sh_pos, sh_mass, maxSharedBodies);

            const float3 dxdt3 = vel_out[blockOffset+i] + dt * dvdt2/2.f;
            const float3 dvdt3 = gravity(pos_out[blockOffset+i] + dt * dxdt2/2.f, sh_pos, sh_mass, maxSharedBodies);

            const float3 dxdt4 = vel_out[blockOffset+i] + dt * dvdt3;
            const float3 dvdt4 = gravity(pos_out[blockOffset+i] + dt * dxdt3, sh_pos, sh_mass, maxSharedBodies);

            pos_out[blockOffset + i] += (dt/6.f) * (dxdt1 + 2.f*(dxdt2 + dxdt3) + dxdt4);
            vel_out[blockOffset + i] += (dt/6.f) * (dvdt1 + 2.f*(dvdt2 + dvdt3) + dvdt4);
        }
    }
}

void integrateRK4(float3* positions, float3* velocities, const float* masses, const float& dt, const int& N) {
    if(device_pos == NULL) {
        cudaMalloc(&device_pos, N*sizeof(float3));
        cudaMalloc(&device_vel, N*sizeof(float3));
        cudaMalloc(&device_masses, N*sizeof(float));

        cudaMalloc(&device_pout, N*sizeof(float3));
        cudaMalloc(&device_vout, N*sizeof(float3));
    }

    cudaMemcpy(device_pos, positions, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_vel, velocities, N*sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_masses, masses, N*sizeof(float), cudaMemcpyHostToDevice);

    const int numBlocks = ceil((float)N / maxSharedBodies);

    rk4_kernel<<<numBlocks, threadsPerBlock>>>(
            device_pos, device_vel, device_masses,
            device_pout, device_vout,
            dt, N);

    cudaThreadSynchronize();

    const cudaError_t error = cudaGetLastError();
    if(error != 0) cerr << "CUDA Error: " << cudaGetErrorString(error) << endl;

    cudaMemcpy(positions, device_pout, N*sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(velocities, device_vout, N*sizeof(float3), cudaMemcpyDeviceToHost);
}
