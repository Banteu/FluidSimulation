#ifndef __KERNEL__
#define __KERNEL__

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thrust\sort.h>
#include <thrust\device_ptr.h>

struct particleData
{
    float3* pos;
    float4* posTextured;
    float3* force;
    float3* vel;
    float3* hVel;
    float3* color;
    float* dens;

    float3* tempVel;
    float3* temphVel;    

    int* hashTableStart;
    int* hashTableEnd;

    float3 gravity;

    int count;
    
    float r;
    float r2;
    float mass;
    float rest_pressure;
    float rest_density;
    float viscosity;
    float pressure_Koeff;


    float maxSpeed;
    float maxAcceleration;
    

    float wallPressure;
    float wallThreshhold;
    float wallDamping;


    float diffKern;
    float pressKern;
    float viscKern;

    int* zind;
    int* pind;

    float3 center;
    float3 sizeContainer;
    unsigned int HASH_TABLE_SIZE;
    float gridDimConst;
};

struct forceData
{
    float power;
    float radius;
    float r2;
    float3 coord;
    float3 velocity;
};


inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if(abort) exit(code);
    }
}
#define gpuErrchk(ans) { gpuAssert(ans, __FILE__, __LINE__);}

void bindToTextures(particleData* pData);

void prepareFluidGPU(particleData pdata, float dt);
void solveFluid(particleData pData, float dt, forceData frc);
void updateSimData(particleData& data);

#endif