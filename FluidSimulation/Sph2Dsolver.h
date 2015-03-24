#pragma once
#include "Headers.h"
#include <vector>

using std::vector;

class Sph2Dsolver
{
public:
    Sph2Dsolver(float x, float y, float w, float h);
    ~Sph2Dsolver(void);


    void drawBoundingBox();
    void createPartiles(uint count ,float volume, float density, float stiffness, float viscosity, float dti);
    void drawParticles();
    void deleteParticles();
    void updateParticle(int id);
    void update();
    void checkBoundary(int id);
    void reset();
    void updateRtParams(float vis, float dtt, float stif);
private:

    uint pCount;
    vec2* pos;
    vec2* vel;

    vec2* pos2;
    vec2* vel2;

    

    vec2* hVel;
    float* dens;

    float density;
    float viscosity;
    float stiffness;
    float volume;
    float mass;
    float r;
    float r2;
    float r3;
    float densKernl;
    float pressKernl;
    float viscKernl;
    

    float cX, cY;
    float width, height;

    float dt;
    

    vector<vector<int> > neighboors;


    vec2 gravity;


    float damping;

    void findAllNeighboors();
    void computeDensities();

};

