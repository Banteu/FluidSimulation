#include "Sph2Dsolver.h"

float myPi = acos(-1.0);
 
Sph2Dsolver::Sph2Dsolver(float x, float y, float w, float h)
{
    cX = x;
    cY = y;
    width = w;
    height = h;
   pCount = 0;
   gravity = vec2(0, -9.8);
   dt = 0.01;
}


void Sph2Dsolver::createPartiles(uint count,float volume, float density, float stiffness, float viscosity, float dti)
{
    pCount = count;
    dt = dti;
    this->volume = volume;
    this->density = density;
    this->stiffness = stiffness;
    this->viscosity = viscosity;

    int activeParticles = 20;

    mass = volume / count * density;
    r = pow(3 * activeParticles * volume / (4 * myPi * count), 1.0f / 3.0f);
    r2 = r * r;
    r3 = r * r * r;
    pos = new vec2[count];
    vel = new vec2[count];
    pos2 = new vec2[count];
    vel2 = new vec2[count];
    hVel = new vec2[count];
    dens = new float[count];

    damping = 1.0f;

    float offset = r * 0.3;
    vec2 ps = vec2(cX - width / 2.0f, cY - height / 2.0f);
    int placed = 0;
    vec2 temp = ps;
    for( int i = 0; placed < count; ++i)
    {
        for(int j = 0; j * j < count && placed < count; ++j)
        {
            pos[placed] = temp;
            vel[placed] = vec2(0, 0);
            hVel[placed] = vec2(0, 0);
            dens[placed] = 0;
            temp.x += offset;
            ++placed;            
        }
        temp.x = ps.x;
        temp.y += offset;         
    }

    densKernl =  315.0 / (64.0 * myPi * r3 * r3 * r3);
    pressKernl = 45.0 / (myPi * r3 * r3);
    viscKernl = 45.0 / (myPi * r3 * r3);

    printf("Simulation params - mass: %f   active radius: %f \n", mass, r);
    printf("Visc kernel: %f \n", viscKernl);

};

void Sph2Dsolver::reset()
{
 float offset = r * 0.3;
    vec2 ps = vec2(cX - width / 2.0f, cY - height / 2.0f);
    int placed = 0;
    vec2 temp = ps;
    for( int i = 0; placed < pCount; ++i)
    {
        for(int j = 0; j * j < pCount && placed < pCount; ++j)
        {
            pos[placed] = temp;
            vel[placed] = vec2(0, 0);
         //   hVel[placed] = vec2(rand() % 100, rand() % 100);
            dens[placed] = 0;
            temp.x += offset;
            ++placed;            
        }
        temp.x = ps.x;
        temp.y += offset;         
    }
}


void Sph2Dsolver::updateRtParams(float vis, float dtt, float stif){
    
    viscosity = vis;
    dt = dtt;
    stiffness = stif;
}

void Sph2Dsolver::drawBoundingBox()
{
    glColor3f(1, 1, 1);
    glBegin(GL_LINE_LOOP);
    glLoadIdentity();
    glVertex3f(cX - width / 2.0, cY - height / 2.0, 0); 
    glVertex3f(cX + width / 2.0, cY - height / 2.0, 0); 
    glVertex3f(cX + width / 2.0, cY + height / 2.0, 0); 
    glVertex3f(cX - width / 2.0, cY + height / 2.0, 0); 
    glVertex3f(cX - width / 2.0, cY - height / 2.0, 0);
    glEnd();
    glFlush();
}

void Sph2Dsolver::computeDensities()
{
    int to = 0;
    vec2 vc;
    float rd;

#pragma omp for
    for (int i = 0; i < pCount; ++i)
    {
        float density = 0;
        for (int j = 0; j < neighboors[i].size(); ++j)
        {
            to = neighboors[i][j];
            vc = pos[i] - pos[to];
            rd = r2 - vc * vc;
            density += rd * rd * rd;
        }   
        dens[i] = densKernl * density * mass + mass;
    }
}

void Sph2Dsolver::drawParticles()
{
    update();
    glPushMatrix();
    for (int i = 0; i < pCount; ++i)
    {
      //  glLoadIdentity();
      //  glTranslatef(pos[i].x, pos[i].y, 0);
      //  glutWireSphere(r, 5, 5);
        glBegin(GL_POINTS);
        glVertex3f(pos[i].x, pos[i].y, 0);
        glEnd();        
    }
    glFlush();
    glPopMatrix();
}
void Sph2Dsolver::checkBoundary(int id)
{
    vec2 normal;
    vec2 oldPos;
    float wDamping = 0.1;
    oldPos = pos[id]; 

    if (pos[id].x > cX + width / 2.0)
    {
        float dt = (pos[id].x - cX - width / 2.0) / hVel[id].x;
        if (dt > 10)
            dt = 0;

        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;
        hVel[id] = hVel[id] - vec2(-1,0) * ((hVel[id] * vec2(-1, 0)) * d);
        vel[id] = vel[id] - vec2(-1,0) * ((vel[id] * vec2(-1, 0)) * d);
        pos[id] += hVel[id] * dt;
    }

    if (pos[id].x < cX - width / 2.0)
    {
        float dt = (-pos[id].x  + cX - width / 2.0) / abs(hVel[id].x);
        if (dt > 10)
            dt = 0;

        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;
        hVel[id] = hVel[id] - vec2(1,0) * ((hVel[id] * vec2(1, 0)) * d);
        vel[id] = vel[id] - vec2(1,0) * ((vel[id] * vec2(1, 0)) * d);

        pos[id] += hVel[id] * dt;
    }

       if (pos[id].y > cY + height / 2.0)
    {
        
        float dt = (pos[id].y - cY - height / 2.0) / hVel[id].y;
        if (dt > 10)
            dt = 0;

        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;

        hVel[id] = hVel[id] - vec2(0,-1) * ((hVel[id] * vec2(0, -1)) * d) ;
        vel[id] = vel[id] - vec2(0,-1) * ((vel[id] * vec2(0, -1)) * d);
        pos[id] += hVel[id] * dt;
    }

    if (pos[id].y < cY - height / 2.0)
    {
  
        float dt = (-pos[id].y + cY - height / 2.0) / abs(hVel[id].y);
        if (dt > 10)
            dt = 0;

        
        pos[id] -= hVel[id] * dt;
        float d = (pos[id] - oldPos).norm() * wDamping / dt / hVel[id].norm() + 1;
        hVel[id] = hVel[id] - vec2(0,1) * ((hVel[id] * vec2(0, 1)) * d);
        vel[id] = vel[id] - vec2(0,1) * ((vel[id] * vec2(0, 1)) * d);
        

        pos[id] += hVel[id] * dt;


    }

}

void Sph2Dsolver::updateParticle(int id)
{
    vec2 acceleration;
    vec2 pressureForce(0, 0);
    vec2 viscosityForce(0, 0);

    int to;
    float cpress = stiffness * (dens[id] - density);
    float express;
    vec2 vc;
    float dst;
    float mult;
    for (int i = 0; i < neighboors[id].size(); ++i)
    {
        to = neighboors[id][i];
        express = stiffness * (dens[to] - density);
        vc = pos[to] - pos[id];
        dst = vc.norm();
        mult = r - dst;
        pressureForce -=  vc * ((cpress + express) * mass * mult * mult * mult / (2 * dst * dens[to]));
        viscosityForce += (vel[to] - vel[id]) * (mult * viscKernl / dens[to]);
    }
    pressureForce = pressureForce * pressKernl;
    viscosityForce = viscosityForce * viscosity * mass;
    acceleration = (pressureForce + viscosityForce) * (1.0f / dens[id]);

//    float accelNorm = acceleration.norm();
 //   if (accelNorm > 3000.0f)
   //    acceleration = acceleration * (100.0f / accelNorm); 

    acceleration += gravity;
    hVel[id] += acceleration * dt;    
    vel2[id]  = hVel[id] + acceleration * dt;
    hVel[id] = hVel[id] * damping;
    vel[id] = vel[id] * damping;
    pos2[id] = pos[id] + hVel[id] * dt;
 

}


Sph2Dsolver::~Sph2Dsolver(void)
{


}

void Sph2Dsolver::update()
{
    findAllNeighboors();
    computeDensities();
    vec2 nr;
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


void Sph2Dsolver::findAllNeighboors()
{
    neighboors.clear();
    neighboors.resize(pCount);
    
    #pragma omp for
    for (int i = 0; i < pCount; ++i)
    {
        vec2 nr;
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

void Sph2Dsolver::deleteParticles()
{
    pCount = 0;
    delete pos;
    delete vel;
    delete pos2;
    delete vel2;
    delete hVel;
    delete dens;
}