#include "SphContainer.h"
#include <time.h>
float MPI = acos(-1.0);

SphContainer::SphContainer(float x, float y, float z, float w, float l, float h)
{    

    containerDrawVertex = new vec3[24];
    containerDrawIndex = new uint[24];

    float dx = x + w / 2;
    float dxm = x - w / 2;
    float dy = y + l / 2;
    float dym = y - l / 2;
    float dz = z + h / 2;
    float dzm = z - h / 2;

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
    centerX = x;
    centerY = y;
    centerZ = z;

    width = w;
    length = l;
    height = h;


}

void SphContainer::createParticles(particleInfo pInfo)
{

    particleCount = pInfo.particleCount;


    rest_density = pInfo.fluidDensity;
    viscosity = pInfo.fluidViscosity;
    radius = pInfo.activeRadius;
    pressure_koeff = pInfo.stiffness;
    r2 = radius * radius;
    r3 = radius * radius * radius;
    mass = 4.0f * r3 * rest_density * MPI / 3.0f / 19;

    float OFFSET = radius * 0.6;




    particlePosition = new vec3[particleCount];
    particleVelocity = new vec4[particleCount];
    particleHvelocity = new vec3[particleCount];
    particleDensity  = new float[particleCount];
    particleColor = new vec3[particleCount];
    particleIndex = new int[particleCount];
    particleZindex = new int[particleCount];


    for (int i = 0; i < particleCount; ++i)
    {
        particleVelocity[i] = vec4(0,0,0,0);
        particleHvelocity[i] = vec3(0,0,0);

        particleDensity[i] = 0;
        particleIndex[i] = i;
        particleZindex[i] = 0;
        particleColor[i] = vec3(1, 1, (rand() % 100) / 500  + 0.05);
    }




    vec3 tempPos = vec3(centerX - width / 2 + OFFSET, centerY - length / 2 + OFFSET,
        centerZ - height / 2 + OFFSET);
    vec3 addPos = tempPos;
    int cnt = 0;
    while(cnt < particleCount)
    {
        for(int i = 0; i * i * i < particleCount && cnt < particleCount; ++i)
        {
            for(int j = 0; j * j * j < particleCount && cnt < particleCount; ++j)
            {
                particlePosition[cnt] = addPos + vec3((rand() % 100) / 1000.0, (rand() % 100) / 1000.0, (rand() % 100) / 1000.0);
                addPos.x += OFFSET;
                ++cnt;
            }
            addPos.y += OFFSET;
            addPos.x = tempPos.x;
        }
        addPos.z += OFFSET;
        addPos.y = tempPos.y;
    }


    glGenBuffers(1, &particlePositionVBO1);
    glBindBuffer(GL_ARRAY_BUFFER, particlePositionVBO1);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(vec3), particlePosition, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenBuffers(1, &particleColorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, particleColorVBO);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(vec3), particleColor, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    gpuErrchk( cudaGraphicsGLRegisterBuffer(&cudaPosVbo, particlePositionVBO1, cudaGraphicsMapFlagsWriteDiscard));
    gpuErrchk( cudaGraphicsGLRegisterBuffer(&cudaColorResource, particleColorVBO, cudaGraphicsMapFlagsWriteDiscard));

    const uint HASH_TABLE_SIZE = 1453021;//prime number;

    particleBeg = new int[HASH_TABLE_SIZE];
    particleEnd = new int[HASH_TABLE_SIZE];


    // gpuErrchk( cudaMalloc(&particlePositionGPU, count * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.vel, particleCount * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.posTextured, particleCount * sizeof(float) * 4));
    gpuErrchk( cudaMalloc(&pData.accel, particleCount * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.hVel, particleCount * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.dens, particleCount * sizeof(float)));
    gpuErrchk(cudaMalloc(&pData.zind, particleCount * sizeof(int)));
    gpuErrchk(cudaMalloc(&pData.pind, particleCount * sizeof(int)));
    gpuErrchk(cudaMalloc(&pData.hashTableStart, HASH_TABLE_SIZE * sizeof(int)));
    gpuErrchk(cudaMalloc(&pData.hashTableEnd, HASH_TABLE_SIZE * sizeof(int)));

    gpuErrchk( cudaMalloc(&pData.tempVel, particleCount * sizeof(float) * 3));
    gpuErrchk( cudaMalloc(&pData.temphVel, particleCount * sizeof(float) * 3));




    // gpuErrchk(cudaMemcpy(particlePositionGPU, particlePosition, count * sizeof(float) * 3, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pData.vel, particleVelocity, particleCount * sizeof(float) * 3, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pData.hVel, particleHvelocity, particleCount * sizeof(float) * 3, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pData.tempVel, particleVelocity, particleCount * sizeof(float) * 3, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pData.temphVel, particleHvelocity, particleCount * sizeof(float) * 3, cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(pData.dens, particleDensity, particleCount * sizeof(float), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(pData.pind, particleIndex, particleCount * sizeof(int), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(pData.zind, particleZindex, particleCount * sizeof(int), cudaMemcpyHostToDevice));


    pData.gravity.x = 0;
    pData.gravity.y = 0;
    pData.gravity.z = -9.8;

    pData.count = particleCount;
    pData.mass = mass;
    pData.r = radius;
    pData.r2 = r2;
    pData.rest_pressure = 0;
    pData.rest_density = rest_density;
    pData.viscosity = viscosity;
    pData.pressure_Koeff = pressure_koeff;
    pData.center.x = centerX;
    pData.center.y = centerY;
    pData.center.z = centerZ;
    pData.sizeContainer.x = width;
    pData.sizeContainer.y = length;
    pData.sizeContainer.z = height;
    pData.gridDimConst = radius * 1.2;

    pData.maxAcceleration = 1000;
    pData.wallDamping = 0.1;

    pData.diffKern = 315.0f / (64.0 * MPI * r3 * r3 * r3);
    pData.pressKern = 45.0 / (MPI * r3 * r3);
    pData.viscKern = 45.0 / (MPI * r3 * r3);

    pData.HASH_TABLE_SIZE = HASH_TABLE_SIZE;

    bindToTextures(&pData);
}

float lstTime = 1;
forceData frc;


void SphContainer::setPower(float power, float rad, vec3 pos, vec3 vel)
{
    frc.coord = make_float3(pos.x,  pos.y, pos.z);
    frc.velocity = make_float3(vel.x,  vel.y, vel.z);
    frc.radius = rad;
    frc.r2 = rad * rad;
    frc.power = power;
}

void SphContainer::computeFluid(float dt)
{
    gpuErrchk( cudaGraphicsMapResources(1, &cudaPosVbo, NULL));
    size_t size;
    gpuErrchk( cudaGraphicsResourceGetMappedPointer((void** )&pData.pos, &size, cudaPosVbo));        
    gpuErrchk( cudaGraphicsMapResources(1, &cudaColorResource, NULL));
    gpuErrchk( cudaGraphicsResourceGetMappedPointer((void** )&pData.color, &size, cudaColorResource));



    
        updateSimData(pData);
        prepareFluidGPU(pData, dt);
        updateSimData(pData);
        solveFluid(pData, dt,frc);



 //   cudaMemcpy(particleZindex, pData.zind, sizeof(int) * particleCount, cudaMemcpyDeviceToHost);
 //   cudaMemcpy(particleIndex, pData.pind, sizeof(int) * particleCount, cudaMemcpyDeviceToHost);

    gpuErrchk( cudaGraphicsUnmapResources(1, &cudaPosVbo, NULL));    
    gpuErrchk( cudaGraphicsUnmapResources(1, &cudaColorResource, NULL));
}






void SphContainer::drawParticles()
{     
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, particlePositionVBO1);
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    glBindBuffer(GL_ARRAY_BUFFER, particleColorVBO);
    glColorPointer(3, GL_FLOAT, 0, NULL);
    glDrawArrays(GL_POINTS, 0, pData.count);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);    
}



void SphContainer::drawContainer()
{

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


}


SphContainer::~SphContainer(void)
{
    delete[] containerDrawIndex;
    delete[] containerDrawVertex; 
}




