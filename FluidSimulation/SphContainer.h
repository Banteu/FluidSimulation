#pragma once
#ifndef __SPH_CONTAINER__
#define __SPH_CONTAINER__

#include "Headers.h"
#include "kernel.cuh"

struct particleInfo
{
    uint particleCount;
    float activeRadius;
    float fluidDensity;
    float fluidViscosity;
    float stiffness; 
    

    float maximumSpeed;
    float maximumAcceleration;


};

class SphContainer
{
public:
    SphContainer(float x, float y, float z, float w, float l, float h);
    void drawContainer();
    void createParticles(particleInfo pInfo);
    void drawParticles();

    void sendDataToGPU();
    void getDataFromGPU();
    void computeFluid(float dt);


    ~SphContainer(void);

private:

    uint particleCount;
    
    particleData pData;

    float viscosity;
    float mass;
    float rest_pressure;
    float rest_density;
    float radius;
    float pressure_koeff;

    float r2;
    float r3;


    vec3* particlePosition;
    vec4* particleVelocity;
    vec3* particleHvelocity;
    float* particleDensity;
    vec3* particleColor;
    int* particleIndex;
    int* particleZindex;
    
    int* particleBeg;
    int* particleEnd;


    cudaGraphicsResource* cudaPosVbo;
    cudaGraphicsResource* cudaColorResource;

    


    uint particlePositionVBO1;
    uint particleColorVBO;
    uint particleMaxCount;
    

    float width;
    float height;
    float length;

    float centerX;
    float centerY;
    float centerZ;

    float divideStepSize;




    vec3* containerDrawVertex;
    uint* containerDrawIndex;
};



#endif