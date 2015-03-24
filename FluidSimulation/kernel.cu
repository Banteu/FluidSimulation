#include "kernel.cuh"
#include <helper_math.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#define DIRECTION_COUNT 27


__constant__ float PI = 3.14159265359;
__constant__ float airFriction = 1.0;

__constant__ int dx[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
__constant__ int dy[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1 };
__constant__ int dz[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1 };


texture<float, 1, cudaReadModeElementType> texDens;
texture<float4, 1, cudaReadModeElementType> texPos;
texture<int, 1, cudaReadModeElementType> texStartHsh;
texture<int, 1, cudaReadModeElementType> texEndHsh;

__constant__ particleData SIM_DATA[1];
__device__ void checkBoundary(int cId);

int blockCount;
int* blockGPUCount;


__device__ float getNorm(float3 obj)
{
    return sqrt(obj.x * obj.x + obj.y * obj.y + obj.z * obj.z);
}
__device__ float getSqNorm(float3 obj)
{
    return obj.x * obj.x + obj.y * obj.y + obj.z * obj.z;
}

void bindToTextures(particleData* pData)
{
    cudaBindTexture(NULL, texDens, pData->dens, pData->count * sizeof(float));    
    cudaBindTexture(NULL, texPos, pData->posTextured, pData->count * sizeof(float) * 4);
    cudaBindTexture(NULL, texStartHsh, pData->hashTableStart, pData->HASH_TABLE_SIZE * sizeof(int));
    cudaBindTexture(NULL, texEndHsh, pData->hashTableEnd, pData->HASH_TABLE_SIZE * sizeof(int));
    cudaMalloc(&blockGPUCount, sizeof(int));
}

__device__ int3 getBlock(float3 pos)
{
    int3 rv;
    rv.x = pos.x / SIM_DATA->gridDimConst;
    rv.y = pos.y / SIM_DATA->gridDimConst;
    rv.z = pos.z / SIM_DATA->gridDimConst;
    return rv;
}

__device__ int getBlockHash(int3 bl)
{
    uint hsh = (((bl.x * 73856093) ^ (bl.y * 19349663) ^ (bl.z * 83492791))) & SIM_DATA->HASH_TABLE_SIZE;
    return hsh;        
}

__device__ unsigned int getParticleHash(float3 bl)
{
    float d = SIM_DATA->gridDimConst;
    uint hsh = ((( (int) (bl.x / d) * 73856093) ^ ( (int) (bl.y / d) * 19349663) ^ ( (int) (bl.z / d) * 83492791)))
        & SIM_DATA->HASH_TABLE_SIZE;         
    return hsh;
}

__global__ void computeDensities()
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= SIM_DATA->count)
        return;

        unsigned int blhsh;
        int3 block, tempBlock;
        int* hashTableStart = SIM_DATA->hashTableStart;
        int* hashTableEnd = SIM_DATA->hashTableEnd;
        int* pInd = SIM_DATA->pind;

        float3 vc, ms;
        float3* pos = SIM_DATA->pos;
        float4 ps4 = tex1Dfetch(texPos, id);
        float3 ps = {ps4.x, ps4.y, ps4.z};
        float rd;
        float r2 = SIM_DATA->r2;
        block = getBlock(ps);
        float densLoc = 0;
        float dtr;
        
        for(int dir = 0; dir < DIRECTION_COUNT; ++dir)
        {
            tempBlock.x = block.x + dx[dir];
            tempBlock.y = block.y + dy[dir];
            tempBlock.z = block.z + dz[dir];
            blhsh = getBlockHash(tempBlock);  
            int strt = tex1Dfetch(texStartHsh, blhsh);
            int end = tex1Dfetch(texEndHsh, blhsh);            
            if (strt >= SIM_DATA->count)
                continue;          
            for (int j = strt; j <= end; ++j)
            {   
               
                ps4 = tex1Dfetch(texPos, j);
                ms.x = ps4.x; ms.y = ps4.y; ms.z = ps4.z;
                vc = ps - ms;
                dtr = vc.x * vc.x + vc.y * vc.y + vc.z * vc.z;
                if (dtr > r2)
                    continue;
                rd = r2 - dtr;
                densLoc += rd * rd * rd;              
            }   
            
        }   
        SIM_DATA->dens[id] = SIM_DATA->diffKern * (densLoc * SIM_DATA->mass);    
}

#define SHARED_SIZE 27 * 30

__global__ void shared_computeDensities()
{
    __shared__ float3 positions[SHARED_SIZE];
    __shared__ int3 baseBlock;
    __shared__ int offset;
   
   
}



__global__ void solveFluid(float dt)
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= SIM_DATA->count)
        return;
    ////// Update particles ////////////////
    
    float3* pos  = SIM_DATA->pos;
    
    float3* vel  = SIM_DATA->vel;
    float3* hVel = SIM_DATA->hVel;
    float* dens  = SIM_DATA->dens;
    int* pInd = SIM_DATA->pind;
    int* hashTableStart = SIM_DATA->hashTableStart;
    int* hashTableEnd = SIM_DATA->hashTableEnd;
    float stiffness = SIM_DATA->pressure_Koeff;
    
    float density = SIM_DATA->rest_density;
    

    float3 acceleration;
    float3 pressureForce;
    float3 viscosityForce;
    pressureForce.x = pressureForce.y = pressureForce.z = 0;
    viscosityForce.x = viscosityForce.y = viscosityForce.z = 0;
    acceleration.x = acceleration.y = acceleration.z = 0;

    float crDens = tex1Dfetch(texDens, cId);
    float cpress = stiffness * (crDens - density);
    float express;
    float3 vc, ms;
    float4 ps4 = tex1Dfetch(texPos, cId);
    float3 ps = {ps4.x, ps4.y, ps4.z};
    float densTo;
    
    
    float3 vlc = vel[cId];
    float dst;
    float mult;
    
    int3 block, tempBlock;
    unsigned int blhsh;

    float r = SIM_DATA->r;
    float r2 = SIM_DATA->r2;
    block = getBlock(ps);    
    
    for(int dir = 0; dir < DIRECTION_COUNT; ++dir)
    {
        tempBlock.x = block.x + dx[dir];
        tempBlock.y = block.y + dy[dir];
        tempBlock.z = block.z + dz[dir];
        blhsh = getBlockHash(tempBlock);
        int strt = tex1Dfetch(texStartHsh, blhsh);
        int end = tex1Dfetch(texEndHsh, blhsh);
        if (strt >= SIM_DATA->count)
            continue;

        for (int j = strt; j <= end ; ++j)
        {              
            if (j == cId)
                continue;
            ps4 = tex1Dfetch(texPos, j);
            densTo = tex1Dfetch(texDens, j);

            ms.x = ps4.x; ms.y = ps4.y; ms.z = ps4.z;
            express = stiffness * (densTo - density);
            vc = ms - ps;
            dst = vc.x * vc.x + vc.y * vc.y + vc.z * vc.z;
            if (dst > r2)
                continue;
            dst = sqrt(dst);
            mult = r - dst;
            pressureForce -=  vc * ((express + cpress) / (2 * dst) * mult * mult * mult);                  
            mult = mult / densTo;
            viscosityForce += (vel[j] - vlc) * mult;            
        } 
    }   


    pressureForce = pressureForce * (SIM_DATA->pressKern * SIM_DATA->mass);
    viscosityForce = viscosityForce * (SIM_DATA->mass * SIM_DATA->viscKern * SIM_DATA->viscosity);    
    acceleration = (pressureForce + viscosityForce) * (1 / crDens);
    float clr = getNorm(acceleration) / 300;
    
    SIM_DATA->color[cId].x = clr;
    SIM_DATA->color[cId].y = clr;
    SIM_DATA->color[cId].z = 0.5 + clr;
    
    acceleration += SIM_DATA->gravity;
    
    hVel[cId] += acceleration * dt;  
    SIM_DATA->force[cId] = acceleration;

}

__global__  void computeZindexes()
{
    
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= SIM_DATA->count)
        return;
    uint pHash = getParticleHash(SIM_DATA->pos[cId]);
    SIM_DATA->zind[cId] = pHash;
    SIM_DATA->pind[cId] = cId;    
}

__global__  void buildHashTable()
{    
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= SIM_DATA->count)
        return;
    uint pHash = SIM_DATA->zind[cId];
    atomicMin(SIM_DATA->hashTableStart + pHash, cId);
    atomicMax(SIM_DATA->hashTableEnd + pHash, cId);   
}

__global__ void updatePositions(float dt)
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= SIM_DATA->count)
        return;
    float4 posOld = tex1Dfetch(texPos, cId);
    float3 pos3 = {posOld.x, posOld.y, posOld.z};
    float3 vl = SIM_DATA->hVel[cId];
    float3 fr = SIM_DATA->force[cId];
    double hdt = dt * 0.5;
    SIM_DATA->vel[cId].x  = vl.x + fr.x * hdt;
    SIM_DATA->vel[cId].y  = vl.y + fr.y * hdt;
    SIM_DATA->vel[cId].z  = vl.z + fr.z * hdt;    
    SIM_DATA->pos[cId] = pos3 + SIM_DATA->hVel[cId] * dt;    
    checkBoundary(cId);
}

__device__ void checkBoundary(int cId)
{
    float3* pos = SIM_DATA->pos + cId;
    float3 center = SIM_DATA->center;
    float3 sizeContainer = SIM_DATA->sizeContainer * 0.5;
    float3* hVel = SIM_DATA->hVel + cId;
    float3* vel = SIM_DATA->vel + cId;
    float wallDamping = SIM_DATA->wallDamping;
    
    if (pos->x > sizeContainer.x + center.x)
    {
        pos->x = sizeContainer.x;
        hVel->x *= -wallDamping;
        vel->x *= -wallDamping;
    }
    if (pos->x < -sizeContainer.x + center.x)
    {
        pos->x = -sizeContainer.x + center.x;
        hVel->x *= -wallDamping;
        vel->x *= -wallDamping;
    }

      if (pos->y > sizeContainer.y + center.y)
    {
        pos->y = sizeContainer.y;
        hVel->y *= -wallDamping;
        vel->y *= -wallDamping;
    }
    if (pos->y < -sizeContainer.y + center.y)
    {
        pos->y = -sizeContainer.y + center.y;
        hVel->y *= -wallDamping;
        vel->y *= -wallDamping;
    }

    if (pos->z > sizeContainer.z + center.z)
    {
        pos->z = sizeContainer.z;
        hVel->z *= -wallDamping;
        vel->z *= -wallDamping;
    }
    if (pos->z < -sizeContainer.z + center.z)
    {
        pos->z = -sizeContainer.z + center.z;
        hVel->z *= -wallDamping;
        vel->z *= -wallDamping;
    }
}

__global__ void initArrays()
{    
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= SIM_DATA->HASH_TABLE_SIZE)
        return;
    SIM_DATA->hashTableStart[cId] = 1000 * 1000 * 1000;
    SIM_DATA->hashTableEnd[cId] = -1;   
}

__global__ void rearrangeParticle()
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= SIM_DATA->count)
        return;

    int ind =  SIM_DATA->pind[cId];
    float3 ps = SIM_DATA->pos[ind];
    SIM_DATA->posTextured[cId].x = ps.x; SIM_DATA->posTextured[cId].y = ps.y; SIM_DATA->posTextured[cId].z = ps.z; SIM_DATA->posTextured[cId].w = 1;
    SIM_DATA->tempVel[cId] = SIM_DATA->vel[ind];
    SIM_DATA->temphVel[cId] = SIM_DATA->hVel[ind];
    SIM_DATA->pind[cId] = cId;
}

__global__ void makeAlignedArray(int* blockCount, int perBlock)
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= SIM_DATA->HASH_TABLE_SIZE)
        return;
    if (SIM_DATA->hashTableStart[cId] >= SIM_DATA->HASH_TABLE_SIZE)
        return;
    
    int count = SIM_DATA->hashTableEnd[cId] - SIM_DATA->hashTableStart[cId] + 1;
    int start = SIM_DATA->hashTableStart[cId];
    int newBlocks = (count + perBlock - 1) / perBlock;

    int pos = atomicAdd(blockCount, newBlocks);
    
    for (int i = 0; i < newBlocks; ++i)
    {
        SIM_DATA->zind[pos + i] = start;
        SIM_DATA->pind[pos + i] = min(perBlock, count);
        start += min(perBlock, count);
        count -= perBlock;
    }
}

__global__ void applyForce(forceData frc)
{
    int cId = blockDim.x * blockIdx.x + threadIdx.x;
    if (cId >= SIM_DATA->count)
        return;


    float4 ps = tex1Dfetch(texPos, cId);
    float3 vc = {ps.x - frc.coord.x, ps.y - frc.coord.y, ps.z - frc.coord.z};


    vc = frc.power * vc / sqrt(dot(vc, vc));
    SIM_DATA->force[cId] += vc;

}



void prepareFluidGPU(particleData pData, float dt)
{
     computeZindexes<<<(pData.count + 255) / 256, 256>>>();  
     thrust::sort_by_key(thrust::device_ptr<int>(pData.zind), thrust::device_ptr<int>(pData.zind) + pData.count, thrust::device_ptr<int>(pData.pind));
     initArrays<<<(pData.HASH_TABLE_SIZE + 255) / 256, 256>>>();
     buildHashTable<<<(pData.count + 255) / 256, 256>>>();
     rearrangeParticle<<<(pData.count + 255) / 256, 256>>>();
     //blockCount = 0;
     //cudaMemcpy(blockGPUCount, &blockCount, sizeof(int), cudaMemcpyHostToDevice);
     //makeAlignedArray<<<(pData.HASH_TABLE_SIZE + 255) / 256, 256>>>(blockGPUCount, 32);
     //cudaMemcpy(&blockCount, blockGPUCount, sizeof(int), cudaMemcpyDeviceToHost);
     
     cudaMemcpy(pData.vel, pData.tempVel, sizeof(float3) * pData.count, cudaMemcpyDeviceToDevice);
     cudaMemcpy(pData.hVel, pData.temphVel, sizeof(float3) * pData.count, cudaMemcpyDeviceToDevice);

}

void solveFluid(particleData pData, float dt, forceData frc)
{
    int threads = 256;
    computeDensities<<<(pData.count + threads - 1) / threads, threads>>>();
    
    //shared_computeDensities<<<blockCount, 32>>>();
    solveFluid<<<(pData.count + threads - 1) / threads, threads>>>(dt);
    applyForce<<<(pData.count + threads - 1) / threads, threads>>>(frc);
    updatePositions<<<(pData.count + threads - 1) / threads, threads>>>(dt);    
}



void updateSimData(particleData& data)
{
    gpuErrchk( cudaMemcpyToSymbol(SIM_DATA, &data, sizeof(data)));
}


