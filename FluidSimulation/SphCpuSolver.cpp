#include "SphCpuSolver.h"
#include <algorithm>


SphCpuSolver::SphCpuSolver(float x, float y, float z, float width, float length, float height)
{
    cX = x;
    cY = y;
    cZ = z;
    this->width = width;
    this->height = height;
    this->length = length;

    pCount = 0;
    pos    = 0;
    vel    = 0;
    pos2   = 0;
    vel2   = 0;
    hVel   = 0;       
    dens   = 0;

    gravity = vec3(0, 0, -19.8);

    containerDrawVertex = new vec3[24];
        containerDrawIndex = new uint[24];
    
        float dx = x + width / 2;
        float dxm = x - width / 2;
        float dy = y + length / 2;
        float dym = y - length / 2;
        float dz = z + height / 2;
        float dzm = z - height / 2;

        containerDrawVertex[0] = vec3(dxm, dym, dzm);
        containerDrawVertex[1] = vec3(dx, dym, dzm);
        containerDrawVertex[2] = vec3(dx, dy, dzm);
        containerDrawVertex[3] = vec3(dxm, dy, dzm);

        containerDrawVertex[4] = vec3(dxm, dym, dz);
        containerDrawVertex[5] = vec3(dxm, dy, dz);
        containerDrawVertex[6] = vec3(dx, dy, dz);
        containerDrawVertex[7] = vec3(dx, dym, dz);


        containerDrawVertex[8] = vec3(dxm, dy, dzm);
        containerDrawVertex[9] = vec3(dx, dy, dzm);
        containerDrawVertex[10] = vec3(dx, dy, dz);
        containerDrawVertex[11] = vec3(dxm, dy, dz);


        containerDrawVertex[12] = vec3(dxm, dym, dzm);
        containerDrawVertex[13] = vec3(dxm, dym, dz);
        containerDrawVertex[14] = vec3(dx, dym, dz);
        containerDrawVertex[15] = vec3(dx, dym, dzm);

        containerDrawVertex[16] = vec3(dx, dym, dzm);
        containerDrawVertex[17] = vec3(dx, dym, dz);
        containerDrawVertex[18] = vec3(dx, dy, dz);
        containerDrawVertex[19] = vec3(dx, dy, dzm);

        containerDrawVertex[20] = vec3(dxm, dym, dzm);
        containerDrawVertex[21] = vec3(dxm, dy, dzm);
        containerDrawVertex[22] = vec3(dxm, dy, dz);
        containerDrawVertex[23] = vec3(dxm, dym, dz);

        for (int i = 0; i < 24; ++i)
        {
            containerDrawIndex[i] = i;
        }
}


SphCpuSolver::~SphCpuSolver(void)
{
    pCount = 0;
    delete[] pos;
    delete[] vel;
    delete[] pos2;
    delete[] vel2;
    delete[] hVel;
    delete[] dens;
    
    pos    = 0;
    vel    = 0;
    pos2   = 0;
    vel2   = 0;
    hVel   = 0;       
    dens   = 0;
}


void SphCpuSolver::findAllNeighboors()
{
    neighboors.clear();
    neighboors.resize(pCount);
    
    #pragma omp for
    for (int i = 0; i < pCount; ++i)
    {
        vec3 nr;
        for (int j = i + 1; j < pCount; ++j)
        {
            nr = pos[j] - pos[i];
            if(nr * nr > r2)
                continue;
            neighboors[i].push_back(j);
            neighboors[j].push_back(i);
        }    
    }
    
}

void SphCpuSolver::drawContainer()
{
   // glDisable(GL_LIGHTING);
    vec3* p = containerDrawVertex;
    for (int i = 0; i < 6; ++i)
    {
        glBegin(GL_LINE_LOOP);
        glVertex3fv((float*)p++);
        glVertex3fv((float*)p++);
        glVertex3fv((float*)p++);
        glVertex3fv((float*)p++);
        glEnd();    
    }

//     glEnable(GL_LIGHTING);
}

float myPi = acos(-1.0);
void SphCpuSolver::createParticles(particleInfo prtInfo)
{

    pCount = prtInfo.particleCount;
    viscosity = prtInfo.fluidViscosity;
    stiffness = prtInfo.stiffness;
    density   = prtInfo.fluidDensity;
    maxSpeed  = prtInfo.maximumSpeed;
    maxAcceleration  = prtInfo.maximumAcceleration;
    r = prtInfo.activeRadius;
    r2 = r * r;
    r3 = r * r * r;
    d = 1.3 * r;

    float onePrtVolume = 4.0 * myPi / 3.0 * r3;
    mass = onePrtVolume * density / 10;


    densKernl = 315.0f / (64.0 * myPi * r3 * r3 * r3);
    pressKernl = 45.0f / (myPi * r3 * r3);
    viscKernl = 45.0f / (myPi * r3 * r3);

    pos = new vec3[pCount];
    pos2 = new vec3[pCount];

    vel = new vec3[pCount];
    vel2 = new vec3[pCount];

    hVel = new vec3[pCount];
    dens = new float[pCount];
    color = new vec3[pCount];
    reset();



}

void SphCpuSolver::reset()
{
    float OFFSET = r ;



    vec3 tempPos = vec3(cX - width / 2 + OFFSET, cY - length / 2 + OFFSET,
        cZ - height / 2 + OFFSET);
    vec3 addPos = tempPos;
    int cnt = 0;
    while(cnt < pCount)
    {
        for(int i = 0; i * i * i < pCount && cnt < pCount; ++i)
        {
            for(int j = 0; j * j * j < pCount && cnt < pCount; ++j)
            {
                pos[cnt] = addPos;
                vel[cnt] = vec3(0, 0, 0);
                hVel[cnt] = vec3(r * (rand() % 100), r * (rand() % 100), r * (rand() % 100));
                color[cnt] = vec3(r * (rand() % 100), r * (rand() % 100), r * (rand() % 100));
                color[cnt].normalize();
                pos2[cnt] = vec3(0, 0, 0);
                vel2[cnt] = vec3(0, 0, 0);

                addPos.x += OFFSET;
                ++cnt;
            }
            addPos.y += OFFSET;
            addPos.x = tempPos.x;
        }
        addPos.z += OFFSET;
        addPos.y = tempPos.y;
    }


}

int dx[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
int dy[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1 };
int dz[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1 };

//int dx[] = {0,1, -1, 0, 0, 0, 0};
//int dy[] = {0,0, 0, 1, -1, 0, 0};
//int dz[] = {0,0, 0, 0, 0, 1, -1};



void SphCpuSolver::computeDensities()
{
    int to = 0;
    vec3 vc;
    float rd;



//#pragma omp for
    for (int i = 0; i < pCount; ++i)
    {
        int blx,bly,blz, blhsh;
        getBlock(i, blx, bly, blz);
        float densLoc = 0;
        float dtr;
        for(int dir = 0; dir < 1; ++dir)
        {
            blhsh = getBlockHash(blx + dx[dir], bly + dy[dir], blz + dz[dir]);  
            int j = blockStart[blhsh];
            int end = blockEnd[blhsh];
            if (j > pCount)
                continue;
            for (; j <= end; ++j)
            {
                to = hashes[j].second;
                vc = pos[i] - pos[to];
                dtr = vc * vc;
                if (dtr > r2)
                    continue;
                rd = r2 - dtr;
                densLoc += rd * rd * rd;
            }   
            
        }   
            dens[i] = densKernl * (densLoc * mass);       
    }
}

void SphCpuSolver::updateParticle(uint id)
{
    vec3 acceleration(0, 0, 0);
    vec3 pressureForce(0, 0, 0);
    vec3 viscosityForce(0, 0, 0);
    int to = 0;

    float cpress = stiffness * (dens[id] - density);
    float express;
    vec3 vc;
    float dst;
    float mult;
    int blx,bly,blz, blhsh;
    getBlock(id, blx, bly, blz);
    for(int dir = 0; dir < 1; ++dir)
    {
        blhsh = getBlockHash(blx + dx[dir], bly + dy[dir], blz + dz[dir]);  
        int j = blockStart[blhsh];
        int end = blockEnd[blhsh];
        if (j > pCount)
            continue;
        for (; j <= end; ++j)
        {
            to = hashes[j].second;
            express = stiffness * (dens[to] - density);
            vc = pos[to] - pos[id];
            dst = vc * vc;
            if (dst > r2 || j == id)
                continue;
            dst = sqrt(dst);
            mult = r - dst;
            pressureForce -=  vc * (((express + cpress) * mult * mult * mult) / (2 * dst));
            viscosityForce += (vel[to] - vel[id]) * (mult / dens[to]);
        }
    }
    pressureForce = pressureForce * (pressKernl * mass);
    viscosityForce = viscosityForce * (mass * viscKernl * viscosity);


    acceleration = (pressureForce + viscosityForce) * (1.0f / dens[id]);


    color[id] = vec3(0.1, 0.1, 0.5) + acceleration.norm() / 200;

    acceleration += gravity;
    hVel[id] += acceleration * dt;    
    vel2[id]  = hVel[id] + (acceleration * (dt * 0.5)) ;
    pos2[id] = pos[id] + hVel[id] * dt;
}

void SphCpuSolver::computeFluid(float dt)
{
    this->dt = dt;
  //  findAllNeighboors();
    buildHashes();
    
    computeDensities();
    #pragma omp for
    for (int i = 0; i < pCount; ++i)
    {
        updateParticle(i);
    }
    std::swap(pos, pos2);
    std::swap(vel, vel2);
    #pragma omp for
    for (int i = 0; i < pCount; ++i)
    {
        checkBoundary(i);
    }

}

void SphCpuSolver::drawParticles()
{
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glColorPointer(3, GL_FLOAT, 0, color);
    glVertexPointer(3, GL_FLOAT, 0, pos);
    
    glDrawArrays(GL_POINTS, 0, pCount);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

}

void SphCpuSolver::checkBoundary(int id)
{
    
    vec3 normal;
    vec3 oldPos;
    float wDamping = 0.1;
    oldPos = pos[id]; 

    if (pos[id].x > cX + width / 2.0)
    {
        float dt = (pos[id].x - cX - width / 2.0) / hVel[id].x;
        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;
        hVel[id] = hVel[id] - vec3(-1, 0, 0) * ((hVel[id] * vec3(-1, 0, 0)) * d);
        vel[id] = vel[id] - vec3(-1, 0, 0) * ((vel[id] * vec3(-1, 0, 0)) * d);
        pos[id] += hVel[id] * dt;
    }

    if (pos[id].x < cX - width / 2.0)
    {
        float dt = (-pos[id].x  + cX - width / 2.0) / abs(hVel[id].x);
        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;
        hVel[id] = hVel[id] - vec3(1, 0, 0) * ((hVel[id] * vec3(1, 0, 0)) * d);
        vel[id] = vel[id] - vec3(1, 0, 0) * ((vel[id] * vec3(1, 0, 0)) * d);

        pos[id] += hVel[id] * dt;
    }

    if (pos[id].y > cY + length / 2.0)
    {

        float dt = (pos[id].y - cY - length / 2.0) / hVel[id].y;
        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;

        hVel[id] = hVel[id] - vec3(0,-1, 0) * ((hVel[id] * vec3(0, -1, 0)) * d) ;
        vel[id] = vel[id] - vec3(0,-1, 0) * ((vel[id] * vec3(0, -1, 0)) * d);
        pos[id] += hVel[id] * dt;
    }

    if (pos[id].y < cY - length / 2.0)
    {

        float dt = (-pos[id].y + cY - length / 2.0) / abs(hVel[id].y);

        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;
        hVel[id] = hVel[id] - vec3(0, 1, 0) * ((hVel[id] * vec3(0, 1, 0)) * d);
        vel[id] = vel[id] - vec3(0, 1, 0) * ((vel[id] * vec3(0, 1, 0)) * d);
        pos[id] += hVel[id] * dt;
    }


    if (pos[id].z > cZ + height / 2.0)
    {

        float dt = (pos[id].z - cZ - height / 2.0) / hVel[id].z;
        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;

        hVel[id] = hVel[id] - vec3(0, 0, -1) * ((hVel[id] * vec3(0, 0, -1)) * d) ;
        vel[id] = vel[id] - vec3(0, 0, -1) * ((vel[id] * vec3(0, 0, -1)) * d);
        pos[id] += hVel[id] * dt;
    }

    if (pos[id].z < cZ - height / 2.0)
    {

        float dt = (-pos[id].z + cZ - height / 2.0) / abs(hVel[id].z);

        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;
        hVel[id] = hVel[id] - vec3(0, 0, 1) * ((hVel[id] * vec3(0, 0, 1)) * d);
        vel[id] = vel[id] - vec3(0, 0, 1) * ((vel[id] * vec3(0, 0, 1)) * d);
        pos[id] += hVel[id] * dt;
    }


}

int SphCpuSolver::getBlock(int pId, int& bx, int& by, int& bz)
{
    bx = pos[pId].x / d;
    by = pos[pId].y / d;
    bz = pos[pId].z / d;
    return 1;
}

uint SphCpuSolver::getBlockHash(int indx, int indy, int indz)
{
    return 1;
   // return (uint)(((indx * 73856093) ^ (indy * 19349663) ^ (indz * 83492791))) % HASH_TABLE_SIZE;         
}

uint SphCpuSolver::pBlochHash(int pId)
{
    return 1;
   // return (uint)((((int)(pos[pId].x / d) * 73856093) ^ ( (int)(pos[pId].y / d) * 19349663) ^ ( (int)(pos[pId].z / d) * 83492791))) % HASH_TABLE_SIZE;
}

void SphCpuSolver::buildHashes()
{
    hashes.clear();
    hashes.resize(pCount);
#pragma omp for
    for (int i = 0; i < pCount; ++i)
    {
        hashes[i].first = pBlochHash(i);
        hashes[i].second = i;
    }

    sort(hashes.begin(), hashes.end());
    int* p1 = blockStart;
    int* p2 = blockEnd;
    for (int i = 0; i < HASH_TABLE_SIZE; ++i)
    {
        *p1 = 1000 * 1000 * 1000;
        *p2 = -1;
        ++p1;
        ++p2;
    }
    for (int i = 0; i < hashes.size(); ++i)
    {
        blockStart[hashes[i].first] = min(blockStart[hashes[i].first], i);
        blockEnd[hashes[i].first] = max(blockEnd[hashes[i].first], i);
    }
}