#include <cstdio>
#include "Headers.h"
#include <time.h>


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

Shader POINT_SHADER; 
Shader SMOOTH_SHADER;
Shader FINAL_RENDER_SHADER;
Shader SKY_BOX_SHADER;
uint mainScreenWidth, mainScreenHeight;
uint textureWidth, textureHeight;
double REDUCTION = 1.0;

Camera mainCamera(0, 0.5, 0.4, 0, 0, -0.3);
Camera TEX_RENDERER_CAMERA(0, 0, 0, 0, 0, -1);

void renderScene(void);
void renderInit(void);

vec3 quadCoord[] = {vec3(-1, -1, -1), vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1)};
vec2 quadTexCoord[] = {vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)};

uint planeVertexBuffer = 0;
uint planeTexCoordBuffer = 0;

uint cubeMapHandler = 0;
byte* checkerTexture;

vec3 deflector(0,0,0);
float deflectorRadius = 0.12;

int oldX = 0;
int oldY = 0;   

vec3* skyBoxVertexArray;
uint* skyBoxIndexArray;


// Program states///////
bool rightClick = false;
bool renderBeauty = true;


float size = 25;
particleInfo prtInf;



SphContainer flSolver(0, 0, 0, 0.7, 0.7, 0.7);

GLenum drbrf[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
Matrix4x4f cameraMtrPrj;
Matrix4x4f cameraMtrMod;
Matrix4x4f prjMtr;
Matrix4x4f modMtr;
Matrix4x4f temp;
float matrForNormals[9];

void generateCheckerTexture()
{
    int size = 256;
    int step = 32;
    checkerTexture = new byte[size * size];

    int color1 = 170;
    int color2 = 80; 

    for (int i = 0; i < size; i += step)
    {
        for (int j = 0; j < size; j += step)
        {
            for (int c = 0; c < step && (i + c) < size; ++ c)
                for (int k = 0; k < step && (j + k) < size; ++k)
                {
                    checkerTexture[ (i + c) * size + j + k] = color1;
                }
                std::swap(color1, color2);
        }
        std::swap(color1, color2);

        color1 = rand() % 250;
        color2 = rand() % 250;

    }
    glEnable(GL_TEXTURE_CUBE_MAP);
    glGenTextures(1, &cubeMapHandler);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapHandler);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R,     GL_CLAMP_TO_EDGE);

    for (int i = 0; i < 6; ++i)
    {
        GLenum trg = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
        glTexImage2D(trg, 0, GL_RED, size, size, 0, GL_RED, GL_UNSIGNED_BYTE, checkerTexture);        
    }
    delete[] checkerTexture;
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
    glDisable(GL_TEXTURE_CUBE_MAP);


}


void buildSkyBox(float x,float y, float z, float w, float h, float l){
    skyBoxVertexArray = new vec3[24];
    skyBoxIndexArray = new uint[24];

    float dx = x + w / 2;
    float dxm = x - w / 2;
    float dy = y + l / 2;
    float dym = y - l / 2;
    float dz = z + h / 2;
    float dzm = z - h / 2;

    skyBoxVertexArray[0] = vec3(dxm, dym, dzm);
    skyBoxVertexArray[1] = vec3(dx, dym, dzm);
    skyBoxVertexArray[2] = vec3(dx, dy, dzm);
    skyBoxVertexArray[3] = vec3(dxm, dy, dzm);

    skyBoxVertexArray[4] = vec3(dxm, dym, dz);
    skyBoxVertexArray[5] = vec3(dxm, dy, dz);
    skyBoxVertexArray[6] = vec3(dx, dy, dz);
    skyBoxVertexArray[7] = vec3(dx, dym, dz);


    skyBoxVertexArray[8] = vec3(dxm, dy, dzm);
    skyBoxVertexArray[9] = vec3(dx, dy, dzm);
    skyBoxVertexArray[10] = vec3(dx, dy, dz);
    skyBoxVertexArray[11] = vec3(dxm, dy, dz);


    skyBoxVertexArray[12] = vec3(dxm, dym, dzm);
    skyBoxVertexArray[13] = vec3(dxm, dym, dz);
    skyBoxVertexArray[14] = vec3(dx, dym, dz);
    skyBoxVertexArray[15] = vec3(dx, dym, dzm);

    skyBoxVertexArray[16] = vec3(dx, dym, dzm);
    skyBoxVertexArray[17] = vec3(dx, dym, dz);
    skyBoxVertexArray[18] = vec3(dx, dy, dz);
    skyBoxVertexArray[19] = vec3(dx, dy, dzm);

    skyBoxVertexArray[20] = vec3(dxm, dym, dzm);
    skyBoxVertexArray[21] = vec3(dxm, dy, dzm);
    skyBoxVertexArray[22] = vec3(dxm, dy, dz);
    skyBoxVertexArray[23] = vec3(dxm, dym, dz);

    for (int i = 0; i < 24; ++i)
    {
        skyBoxIndexArray[i] = i;
    }
}

void drawSkyBox()
{
    glEnable(GL_TEXTURE_CUBE_MAP);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapHandler);
    glColor3f(1, 1, 1);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, skyBoxVertexArray);
    glDrawElements(GL_QUADS, 24, GL_UNSIGNED_INT, skyBoxIndexArray);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_TEXTURE_CUBE_MAP);
}

void mouseClickHandler(int button, int state, int x, int y)
{
    oldX = x;
    oldY = y;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
        rightClick = true;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
        rightClick = false;
};

void keyboardPressHandler(unsigned char key, int x, int y)
{
    if (key == 'r')
        renderBeauty = !renderBeauty;

    }

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


void setFramebufferOutputs();


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

    glBindBuffer(GL_ARRAY_BUFFER, planeVertexBuffer);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, planeTexCoordBuffer);
    glTexCoordPointer(2, GL_FLOAT, 0, 0);
    glDrawArrays(GL_QUADS, 0, 4);     
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glActiveTexture(GL_TEXTURE0);
    glDisable(GL_TEXTURE_2D);
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
}

vec4 multVecOnMtr(vec4& a, Matrix4x4f& mt)
{
    vec4 retVal;
    float* dt = mt.getDataPointer();

    retVal.x = a.x * dt[0] + a.y * dt[4] + a.z * dt[8] + a.w * dt[12];
    retVal.y = a.x * dt[1] + a.y * dt[5] + a.z * dt[9] + a.w * dt[13];
    retVal.z = a.x * dt[2] + a.y * dt[6] + a.z * dt[10] + a.w * dt[14];
    retVal.w = a.x * dt[3] + a.y * dt[7] + a.z * dt[11] + a.w * dt[15];
    return retVal;
}


void mouseHandler(int x, int y)
{

    float dx = (oldX - x) / 60.0;
    float dy = (oldY - y) / 60.0;

    if (rightClick)
    {
        mainCamera.setRenderMatrix();
        glGetFloatv(GL_MODELVIEW_MATRIX, cameraMtrMod.getDataPointer());
        glGetFloatv(GL_PROJECTION_MATRIX, cameraMtrPrj.getDataPointer());
        vec4 defl = vec4(deflector);
        defl.w = 1;
        Matrix4x4f tmp = cameraMtrMod * cameraMtrPrj;
        vec4 viewSpacePos = multVecOnMtr(defl, tmp);
        viewSpacePos = viewSpacePos * (1.0 / viewSpacePos.w);
        viewSpacePos.x = (float)x / mainScreenWidth;
        viewSpacePos.x = viewSpacePos.x * 2 - 1;
        viewSpacePos.y = 1 - (float)y / mainScreenHeight;
        viewSpacePos.y = viewSpacePos.y * 2 - 1;
        viewSpacePos = multVecOnMtr(viewSpacePos, tmp.getInversed());
        viewSpacePos = viewSpacePos * (1.0 / viewSpacePos.w);
        vec3 vel = viewSpacePos.getVec3() - deflector;
        deflector = viewSpacePos.getVec3();
        flSolver.setPower(1, deflectorRadius, deflector, vel * 10.1);
        oldX = x;
        oldY = y;
        return;
    }
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

    mainScreenHeight = 1024;
    mainScreenWidth = 1024;
    glutInitWindowSize(mainScreenWidth,mainScreenHeight);
    textureHeight = mainScreenHeight / REDUCTION;
    mainScreenWidth = mainScreenWidth / REDUCTION;
    glutCreateWindow("Fluid simulation");
    glutDisplayFunc(renderScene);
    glutReshapeFunc(changeSize);
    glutMotionFunc(mouseHandler);
    glutMouseFunc(mouseClickHandler);
    glutKeyboardFunc(keyboardPressHandler);   
    glewInit();
    generateCheckerTexture();
    TEX_RENDERER_CAMERA.setOrthoBase(1);
    glColor3f(1,1,1);
    glClearColor(0, 0, 0, 0);
    buildSkyBox(0, 0, 0, 10, 10, 10);
    renderInit();



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

    return 0;
}

int main(int argc, char **argv)
{

    LOG_FILE_POINTER = stdout;
    TEX_RENDERER_CAMERA.setProjectionType(ORTHO_PROJECTION);
    prtInf.particleCount = 35536;
    prtInf.activeRadius = 0.024;
    prtInf.fluidDensity = 1000.0f;
    prtInf.fluidViscosity = 2.5f;
    prtInf.stiffness = 2.5f;      

    createWindow(argc, argv);   
    return 0;
}   

void renderInit()
{

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_COLOR, GL_DST_COLOR);
    glBlendEquation(GL_FUNC_SUBTRACT);

    POINT_SHADER.createShader("shader/POINT_VX.vs", "shader/POINT_FS.fs", "");
    SMOOTH_SHADER.createShader("shader/SMOOTH_VX.vs", "shader/SMOOTH_FS.fs", "");
    FINAL_RENDER_SHADER.createShader("shader/FINAL_RENDER_SHADER_VS.vs", "shader/FINAL_RENDER_SHADER_FS.fs", "");
    SKY_BOX_SHADER.createShader("shader/sky_box_shader.vs", "shader/sky_box_shader.fs", "");

    glEnableVertexAttribArray(1);
    glGenBuffers(1, &planeVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, planeVertexBuffer);
    glVertexAttribPointer(1, 3, GL_FLOAT, 0, 0, 0);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(vec3), quadCoord, GL_STATIC_DRAW);

    glEnableVertexAttribArray(2);
    glGenBuffers(1, &planeTexCoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, planeTexCoordBuffer);
    glVertexAttribPointer(2, 2, GL_FLOAT, 0, 0, 0);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(vec2), quadTexCoord, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);  
    glPointSize(5);

    CHECK_ERRORS;
}

void renderScene(void) {

    clock_t tm1 = clock();
    flSolver.computeFluid(0.002);  




    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
    mainCamera.setRenderMatrix(); 
    glGetFloatv(GL_MODELVIEW_MATRIX, cameraMtrMod.getDataPointer());
    glGetFloatv(GL_PROJECTION_MATRIX, cameraMtrPrj.getDataPointer());   


    if (renderBeauty)
    {
        glViewport(0, 0, textureWidth, textureHeight);
        POINT_SHADER.assignShader();    
        POINT_SHADER.sendViewMatrices(cameraMtrPrj.getDataPointer(), cameraMtrMod.getDataPointer());
        std::swap(RENDERER.depthTexture1, RENDERER.depthTexture2);
        std::swap(RENDERER.fluidDepthTexture1, RENDERER.fluidDepthTexture2);
        setFramebufferOutputs();
        glBindFramebufferEXT(GL_FRAMEBUFFER, RENDERER.framebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
        POINT_SHADER.sendInt("drawingPass", 1);
    }

    flSolver.drawParticles(); 

    // Depth rendering ///
    //   glDisable(GL_DEPTH_TEST);
    //   glEnable(GL_BLEND);
    //   glFramebufferTextureEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, RENDERER.fluidDepthTexture2, 0);
    //   POINT_SHADER.sendInt("drawingPass", 2);
    //       flSolver.drawParticles(); 
    //       glDisable(GL_BLEND);
    //       glEnable(GL_DEPTH_TEST);
    //////////////////////

    if (renderBeauty)
    {
        CHECK_ERROR;    
        SMOOTH_SHADER.assignShader();
        std::swap(RENDERER.depthTexture1, RENDERER.depthTexture2);
        std::swap(RENDERER.fluidDepthTexture1, RENDERER.fluidDepthTexture2);
        setFramebufferOutputs();
        glBindFramebufferEXT(GL_FRAMEBUFFER, RENDERER.framebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        TEX_RENDERER_CAMERA.setRenderMatrix();
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
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapHandler);

        FINAL_RENDER_SHADER.assignShader(); 
        FINAL_RENDER_SHADER.sendViewMatrices(prjMtr.getDataPointer(), modMtr.getDataPointer());
        FINAL_RENDER_SHADER.sendMtr4x4Data("oldProjection_matrix", 1, cameraMtrPrj.getDataPointer());
        FINAL_RENDER_SHADER.sendMtr4x4Data("oldModelview_matrix", 1, cameraMtrMod.getDataPointer());
        temp = (cameraMtrPrj).getInversed();
        FINAL_RENDER_SHADER.sendMtr4x4Data("inverted_matrix", 1, temp.getDataPointer()); 

        temp = (cameraMtrMod * cameraMtrPrj).getInversed();
        FINAL_RENDER_SHADER.sendMtr4x4Data("toWorldMatrix", 1, temp.getDataPointer()); 

        temp = (cameraMtrMod).getInversed();
        temp.getMainMinor(matrForNormals);
        FINAL_RENDER_SHADER.sendMtr3x3Data("normalToWorld", 1, matrForNormals); 


        FINAL_RENDER_SHADER.sendCameraPosition(&mainCamera);
        renderTextureOnScreen();
        glUseProgram(0);   
        glDisable(GL_TEXTURE_CUBE_MAP);
        glDisable(GL_TEXTURE_2D);
    }

    mainCamera.setRenderMatrix();
    glColor3f(1, 1, 1); 
    SKY_BOX_SHADER.assignShader();
    SKY_BOX_SHADER.sendViewMatrices(cameraMtrPrj.getDataPointer(), cameraMtrMod.getDataPointer());

    glActiveTexture(GL_TEXTURE0);
    drawSkyBox();
    glUseProgram(0);

    flSolver.drawContainer();


    temp = getTranslateMatrix(deflector.x, deflector.y, deflector.z);

    glLoadIdentity();
    glTranslatef(deflector.x, deflector.y, deflector.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, temp.getDataPointer());
    temp = temp * cameraMtrMod;
    glLoadMatrixf(temp.getDataPointer());
    glColor3f(1, 0, 0);
    glutWireSphere(deflectorRadius, 20, 20);
    glColor3f(1, 1, 1);

    glFlush();
    glutSwapBuffers();
    clock_t tm2 = clock();
    printf("FPS: %f \n", 1.0 / (((double) tm2 - tm1) / CLOCKS_PER_SEC)); 
}




