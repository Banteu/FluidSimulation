#include <cstdio>
#include "Headers.h"

Shader POINT_SHADER; 
Shader SMOOTH_SHADER;
Shader FINAL_RENDER_SHADER;

uint mainScreenWidth, mainScreenHeight;
uint textureWidth, textureHeight; // twice lower resolution

Camera mainCamera(0, 0.5, 0.4, 0, 0, -0.3);
Camera TEX_RENDERER_CAMERA(0, 0, 0, 0, 0, -1);
void renderScene(void);
void renderInit(void);

vec3 quadCoord[] = {vec3(-1, -1, -1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1)};
vec2 quadTexCoord[] = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};

uint scrVaoHandle = 0;
uint scrVerBuffer = 0;
uint scrTexBuffer = 0;

uint cubeMapEnv = 0;

byte* checkerTextureColor;


void generateCheckerTexture()
{
    int size = 16;
    int step = 1;
    checkerTextureColor = new byte[size * size];

    int color1 = 40;
    int color2 = 128; 

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            checkerTextureColor[i * size + j] = color1;
            std::swap(color1, color2);
        }
        glEnable(GL_TEXTURE_CUBE_MAP);
        glGenTextures(1, &cubeMapEnv);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapEnv);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R,     GL_CLAMP_TO_EDGE);

        for (int i = 0; i < 6; ++i)
        {
            GLenum trg = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
            glTexImage2D(trg, 0, GL_RED, size, size, 0, GL_RED, GL_UNSIGNED_BYTE, checkerTextureColor);        
        }
        delete[] checkerTextureColor;
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        glDisable(GL_TEXTURE_CUBE_MAP);


}

vec3* containerDrawVertex;
uint* containerDrawIndex;
void buildCube(float x,float y, float z, float w, float h, float l){
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
}

void drawCube()
{
    glColor3f(1, 1, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, containerDrawVertex);
    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_INT, containerDrawIndex);
    glDisableClientState(GL_VERTEX_ARRAY);

}


float* texturetst = 0;


void CHECK_ERRORS(int line)
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        printf("OPENGL ERROR: FILE %s, LINE: %d", __FILE__, line);
        exit(-1);
    }

}

void CHECK_FRAMEBUFFER_ERRS(int line)
{
    GLenum error = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER);
    if (error != GL_FRAMEBUFFER_COMPLETE)
    {
        printf("FRAMEBUFFER INCOMPLETE: FILE %s, LINE: %d", __FILE__, line);
        exit(-1);
    }

}

#define CHECK_ERROR CHECK_ERRORS(__LINE__)
#define CHECK_FRAMEBUFFER CHECK_FRAMEBUFFER_ERRS(__LINE__);



struct RenderSystem
{
    uint framebuffer;
    uint depthBuffer;
    
    uint depthTexture1;    
    uint fluidDepthTexture1;
    
    uint depthTexture2;    
    uint fluidDepthTexture2;
    

    uint normalTexture;

} RENDERER;

float size = 25;
particleInfo prtInf;

//SphCpuSolver flSolver(0, 0, 0, 1, 1, 1);

SphContainer flSolver(0, 0, 0, 0.7, 0.7, 0.7);

extern "C" _declspec(dllexport) void resetSimulation()
{    
} 


extern "C" _declspec(dllexport) void updateRealTimeParams(float viscosity, float dt, float stiffness)
{
}

extern "C" _declspec(dllexport) void createPrts(uint count, float volume, float viscosity,
                                                float stifness, float density, float dti)
{
    printf("Count: %d volume: %f viscosity: %f stifness: %f density: %f time step: %f", count, volume, viscosity,
        stifness, density, dti);

}

void setFramebufferOutputs();

GLenum drbrf[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};

void generateBuffers()
{
    glEnable(GL_TEXTURE_2D);
    
    
    glGenTextures(1, &RENDERER.depthTexture1);
    glBindTexture(GL_TEXTURE_2D, RENDERER.depthTexture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenTextures(1, &RENDERER.depthTexture2);
    glBindTexture(GL_TEXTURE_2D, RENDERER.depthTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
    
    glGenTextures(1, &RENDERER.fluidDepthTexture1);
    glBindTexture(GL_TEXTURE_2D, RENDERER.fluidDepthTexture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        
    glGenTextures(1, &RENDERER.fluidDepthTexture2);
    glBindTexture(GL_TEXTURE_2D, RENDERER.fluidDepthTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);

    CHECK_ERROR;

    glGenFramebuffersEXT(1, &RENDERER.framebuffer);
    
    glGenRenderbuffersEXT(1, &RENDERER.depthBuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER, RENDERER.depthBuffer);
    glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, textureWidth, textureHeight);
    glBindRenderbufferEXT(GL_RENDERBUFFER, 0);

    /*
    glGenRenderbuffersEXT(1, &RENDERER.computedDepthBuffer);
    glGenRenderbuffersEXT(1, &RENDERER.normalBuffer);    
    glBindRenderbufferEXT(GL_RENDERBUFFER, RENDERER.normalBuffer);    
    glBindRenderbufferEXT(GL_RENDERBUFFER, RENDERER.fluidDepthBuffer);    
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, RENDERER.normalBuffer);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_RENDERBUFFER, RENDERER.fluidDepthBuffer);
    */

    glBindFramebufferEXT(GL_FRAMEBUFFER, RENDERER.framebuffer);
    
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RENDERER.depthBuffer);
    

    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, RENDERER.depthTexture2, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, RENDERER.fluidDepthTexture2, 0);
    CHECK_FRAMEBUFFER;
    
    glDrawBuffers(2, drbrf);


    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
    
    
    CHECK_ERROR;

    glDisable(GL_TEXTURE_2D);

}

void setFramebufferOutputs()
{
    glBindFramebufferEXT(GL_FRAMEBUFFER, RENDERER.framebuffer);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RENDERER.depthBuffer);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, RENDERER.depthTexture2, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, RENDERER.fluidDepthTexture2, 0);

    CHECK_FRAMEBUFFER;
    glDrawBuffers(2, drbrf);

    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

}



void renderTextureOnScreen()
{
    
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
   

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RENDERER.depthTexture1);
    

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, RENDERER.fluidDepthTexture1);
    

    glBindBuffer(GL_ARRAY_BUFFER, scrVerBuffer);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, scrTexBuffer);
    glTexCoordPointer(2, GL_FLOAT, 0, 0);
    glDrawArrays(GL_QUADS, 0, 4);  
    

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisable(GL_TEXTURE_2D);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}


void deleteBuffers()
{
    glDeleteTextures(1, &RENDERER.fluidDepthTexture1);
    glDeleteTextures(1, &RENDERER.depthTexture1);
    

    glDeleteTextures(1, &RENDERER.fluidDepthTexture2);
    
    glDeleteTextures(1, &RENDERER.depthTexture2);
    
    glDeleteRenderbuffersEXT(1, &RENDERER.depthBuffer);
    glDeleteFramebuffersEXT(1, &RENDERER.framebuffer);

  

}

double REDUCTION = 1.0;


void changeSize(int w, int h) {
    
    mainScreenWidth = w;
    mainScreenHeight = h;    

    textureHeight = mainScreenHeight / REDUCTION;
    textureWidth = mainScreenWidth / REDUCTION;
    
    
    SMOOTH_SHADER.sendFloat("texSizeX", 1.0f / textureWidth);
    SMOOTH_SHADER.sendFloat("texSizeY", 1.0f / textureHeight);

    FINAL_RENDER_SHADER.sendFloat("texSizeX", 1.0f / mainScreenWidth);
    FINAL_RENDER_SHADER.sendFloat("texSizeY", 1.0f / mainScreenHeight);

    
    SMOOTH_SHADER.sendInt("tSam", 0);
    SMOOTH_SHADER.sendInt("deepSam", 1);

    FINAL_RENDER_SHADER.sendInt("tSam", 0);
    FINAL_RENDER_SHADER.sendInt("deepSam", 1);
    FINAL_RENDER_SHADER.sendInt("envir", 2);

    glUseProgram(0);

    glViewport(0,0,mainScreenWidth,mainScreenHeight);    
    mainCamera.setAspectRatio((float)w / h);
    mainCamera.setRenderMatrix();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    deleteBuffers();
    generateBuffers();
    delete[] texturetst;
         texturetst = new float[textureHeight * textureWidth];
     for (int i = 0; i < textureHeight * textureWidth; ++i)
         texturetst[i] = (rand() % 1000) / 1000.0;

}

int oldX = 0;
int oldY = 0;   

void mouseHandler(int x, int y)
{
    float dx = (oldX - x) / 60.0;
    float dy = (oldY - y) / 60.0;
    mainCamera.rotateAroundAim(dx, dy);
    oldX = x;
    oldY = y;
    renderScene();
}



bool createWindow(int argc, char **argv)
{
    glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(1024,1024);
    mainScreenHeight = 1024;
    mainScreenWidth = 1024;
    textureHeight = mainScreenHeight / REDUCTION;
    mainScreenWidth = mainScreenWidth / REDUCTION;

	glutCreateWindow("Fluid simulation");
    glutDisplayFunc(renderScene);
    glutReshapeFunc(changeSize);
    glColor3f(1,1,1);
    glClearColor(0, 0, 0, 0);
    glutMotionFunc(mouseHandler);
  
    glewInit();
    generateCheckerTexture();
    buildCube(0, 0, 0, 100, 100, 100);
    
     if (! glewIsSupported("GL_VERSION_2_0 "))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

     CHECK_ERRORS;

     renderInit();
         
    //glEnable(GL_BLEND);
    //glBlendEquation(GL_FUNC_ADD);
    //glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);



    ///// CUDA accelerator initialization ////
    cudaDeviceProp properties;
    cudaSetDevice(0);
    cudaGLSetGLDevice(0);
    cudaGetDeviceProperties(&properties, 0);
    printf("CUDA Accelerator: %s \n", properties.name);
    printf("Maximum threads dim count: %d %d %d \n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
    printf("Maximum grid size: %d %d %d \n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);
    printf("Maximum memory size: %d \n", properties.totalGlobalMem / 1024 / 1024);
    printf("Maximum threads per block: %d \n", properties.maxThreadsPerBlock);
    glutIdleFunc(renderScene);

    

    flSolver.createParticles(prtInf);

	glutMainLoop();
}

int main(int argc, char **argv)
{
    LOG_FILE_POINTER = stdout;
    TEX_RENDERER_CAMERA.setProjectionType(ORTHO_PROJECTION);

    
    prtInf.particleCount = 40000;
    prtInf.activeRadius = 0.012;
    prtInf.fluidDensity = 1000.0f;
    prtInf.fluidViscosity = 1.5f;
    prtInf.stiffness = 2.0f;      
    
    createWindow(argc, argv);
	return 1;
}   

void renderInit()
{
         
     glEnable(GL_PROGRAM_POINT_SIZE);
     glEnable(GL_POINT_SPRITE);
     glEnable(GL_DEPTH_TEST);
     //glEnable(GL_BLEND);
   //  glBlendFunc(GL_SRC_COLOR, GL_DST_COLOR);
    // glBlendEquation(GL_FUNC_ADD);
     


     POINT_SHADER.createShader("shader/POINT_VX.vs", "shader/POINT_FS.fs", "");
     SMOOTH_SHADER.createShader("shader/SMOOTH_VX.vs", "shader/SMOOTH_FS.fs", "");
     FINAL_RENDER_SHADER.createShader("shader/FINAL_RENDER_SHADER_VS.vs", "shader/FINAL_RENDER_SHADER_FS.fs", "");
    

     glEnableVertexAttribArray(1);
     glGenBuffers(1, &scrVerBuffer);
     glBindBuffer(GL_ARRAY_BUFFER, scrVerBuffer);
     glVertexAttribPointer(1, 3, GL_FLOAT, 0, 0, 0);
     glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(vec3), quadCoord, GL_STATIC_DRAW);


     glEnableVertexAttribArray(2);
     glGenBuffers(1, &scrTexBuffer);
     glBindBuffer(GL_ARRAY_BUFFER, scrTexBuffer);
     glVertexAttribPointer(2, 2, GL_FLOAT, 0, 0, 0);
     glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(vec2), quadTexCoord, GL_STATIC_DRAW);

     glBindBuffer(GL_ARRAY_BUFFER, 0);  

     CHECK_ERRORS;
}

Matrix4x4f cameraMtrPrj;
Matrix4x4f cameraMtrMod;

Matrix4x4f prjMtr;
Matrix4x4f modMtr;
Matrix4x4f temp;

void renderScene(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
    mainCamera.setRenderMatrix(); 
    glGetFloatv(GL_MODELVIEW_MATRIX, cameraMtrMod.getDataPointer());
    glGetFloatv(GL_PROJECTION_MATRIX, cameraMtrPrj.getDataPointer());
    glViewport(0, 0, textureWidth, textureHeight);
    flSolver.computeFluid(0.001);  

    
        std::swap(RENDERER.depthTexture1, RENDERER.depthTexture2);
    std::swap(RENDERER.fluidDepthTexture1, RENDERER.fluidDepthTexture2);
    setFramebufferOutputs();

    glBindFramebufferEXT(GL_FRAMEBUFFER, RENDERER.framebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
    flSolver.drawParticles();      

    
    SMOOTH_SHADER.assignShader();
    std::swap(RENDERER.depthTexture1, RENDERER.depthTexture2);
    std::swap(RENDERER.fluidDepthTexture1, RENDERER.fluidDepthTexture2);
    setFramebufferOutputs();
    glBindFramebufferEXT(GL_FRAMEBUFFER, RENDERER.framebuffer);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    TEX_RENDERER_CAMERA.setRenderMatrix();
    TEX_RENDERER_CAMERA.setOrthoBase(1);
    glGetFloatv(GL_MODELVIEW_MATRIX, modMtr.getDataPointer());
    glGetFloatv(GL_PROJECTION_MATRIX, prjMtr.getDataPointer());
    SMOOTH_SHADER.sendViewMatrices(prjMtr.getDataPointer(), modMtr.getDataPointer()); 
        

    renderTextureOnScreen();
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
   
    
    std::swap(RENDERER.depthTexture1, RENDERER.depthTexture2);
    std::swap(RENDERER.fluidDepthTexture1, RENDERER.fluidDepthTexture2);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
    glViewport(0, 0, mainScreenWidth, mainScreenHeight); 

    glEnable(GL_TEXTURE_CUBE_MAP);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapEnv);

    FINAL_RENDER_SHADER.assignShader(); 
    FINAL_RENDER_SHADER.sendViewMatrices(prjMtr.getDataPointer(), modMtr.getDataPointer());
    FINAL_RENDER_SHADER.sendMtr4x4Data("oldProjection_matrix", 1, cameraMtrPrj.getDataPointer());
    FINAL_RENDER_SHADER.sendMtr4x4Data("oldModelview_matrix", 1, cameraMtrMod.getDataPointer());
    temp = (prjMtr).getInversed();
    FINAL_RENDER_SHADER.sendMtr4x4Data("inverted_matrix", 1, temp.getDataPointer()); 
    FINAL_RENDER_SHADER.sendCameraPosition(&mainCamera);
    renderTextureOnScreen();
    glUseProgram(0);
    
    glDisable(GL_TEXTURE_CUBE_MAP);


    mainCamera.setRenderMatrix(); 
    // drawCube();
    glColor3f(1, 1, 1);
    flSolver.drawContainer();
    glFlush();
	glutSwapBuffers();
}
