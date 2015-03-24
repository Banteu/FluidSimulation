#pragma once


#include "Headers.h"
#include <vector>
using std::vector;
using std::pair;

const int HASH_TABLE_SIZE = 1000 * 100;

struct particleInfo;


class SphCpuSolver
{
public:
    SphCpuSolver(float x, float y, float z, float width, float length, float height);
    ~SphCpuSolver(void);

    void createParticles(particleInfo pInfo);
    void drawContainer();
    void drawParticles();
    void reset();
    void computeFluid(float dt);

private:

    void updateParticle(uint id);
    void checkBoundary(int id);
    void buildHashes();
    

    uint getBlockHash(int bx, int by, int bz);
    uint pBlochHash(int pId);
    int getBlock(int pId, int& bx, int& by, int& bz);

    float scale;




    uint pCount;
    vec3* pos;
    vec3* vel;
    vec3* pos2;
    vec3* vel2;
    vec3* hVel;    
    vec3* color;
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

    float maxSpeed;
    float maxAcceleration;
    

    float cX, cY, cZ;
    float width, height, length;

    float dt;
    

    vector<vector<int> > neighboors;
    vector<pair<uint, int> > hashes;
    int blockStart[HASH_TABLE_SIZE];
    int blockEnd[HASH_TABLE_SIZE];
    float d;

    vec3 gravity;


    float wallDamping;

    void findAllNeighboors();
    void computeDensities();


    vec3* containerDrawVertex;
    uint* containerDrawIndex;

};

