#include "Material.h"

const int SHADER_CODEBUFFER_SIZE = 15000;

FILE* LOG_FILE_POINTER = 0;

Shader::Shader()
{
    geometryShader = 0;
    vertexShader = 0;
    fragmentShader = 0;
    geometryShaderCodeBuffer = 0;
    geometryShaderSize = 0;
    vertexShaderCodeBuffer = 0;
    vertexShaderSize = 0;
    fragmentShaderCodeBuffer = 0;
    fragmentShaderSize = 0;
    shaderProgram = 0;
}
Shader::~Shader()
{

}

uint Shader::getProgram()
{
    return shaderProgram;
}

void Shader::sendFloat(const char* name, float val){
    assignShader();
    uint unifLocat = glGetUniformLocation(shaderProgram, name);
    glUniform1f(unifLocat, val);
};

void Shader::sendInt(const char* name, int val){
    assignShader();
    uint unifLocat = glGetUniformLocation(shaderProgram, name);
    glUniform1i(unifLocat, val);
};


void Shader::createShader(const string& vertexShader, const string& fragmentShader, const string& geometryShader)
{
    if(vertexShader.empty() == false)
    {
        readVertexShader(vertexShader);    
    }
    if(fragmentShader.empty() == false)
    {
        readFragmentShader(fragmentShader);    
    }
    if(geometryShader.empty() == false)
    {
        readGeometryShader(geometryShader);    
    }
    createShaderProgram();
    
};

void Shader::getCode(const string& filename, char** saveTo, int& count)
{

    FILE* shaderCode = fopen(filename.c_str(),"r");
    if(shaderCode == 0)
    {
        PRINT_LOG("Can't open file");
        return;
    }
    char* tempShaderCodeBuffer = new char[SHADER_CODEBUFFER_SIZE];

    int fileSize = fread(tempShaderCodeBuffer, sizeof(char), SHADER_CODEBUFFER_SIZE, shaderCode);
    
    count = fileSize;
    (*saveTo) = new char[fileSize];
    memcpy((*saveTo), tempShaderCodeBuffer, sizeof(char) * fileSize);

    delete[] tempShaderCodeBuffer;
}

void Shader::readVertexShader(const string& filename)
{
    getCode(filename, &vertexShaderCodeBuffer, vertexShaderSize);
    vertexShader = glCreateShaderObjectARB(GL_VERTEX_SHADER);
    glShaderSourceARB(vertexShader, 1, (const GLcharARB**)&vertexShaderCodeBuffer, &vertexShaderSize);
    glCompileShaderARB(vertexShader);

    /////Debug////
    PRINT_LOG("____VERTEX_SHADER_LOG_____ ");
    int logLen;
    char logBuf[1024];
    glGetInfoLogARB(vertexShader, 1024,&logLen,logBuf);
    PRINT_LOG(logBuf);
}
void Shader::readFragmentShader(const string& filename)
{
    
    getCode(filename, &fragmentShaderCodeBuffer, fragmentShaderSize);
    fragmentShader = glCreateShaderObjectARB(GL_FRAGMENT_SHADER);
    glShaderSourceARB(fragmentShader, 1,(const GLcharARB**) &fragmentShaderCodeBuffer, &fragmentShaderSize);
    glCompileShaderARB(fragmentShader);


    /////Debug////
    PRINT_LOG("____FRAGMENT_SHADER_LOG_____ ");
    int logLen;
    char logBuf[1024];
    glGetInfoLogARB(fragmentShader, 1024,&logLen,logBuf);
    PRINT_LOG(logBuf);
}

void Shader::sendDataBlock(int location, void* data)
{
    
}
void Shader::readGeometryShader(const string& filename)
{

}

void Shader::sendViewMatrices(float* projection, float* modelview)
{
    this->assignShader();
    uint projectionLocation = glGetUniformLocation(shaderProgram, "current_projection_matrix");
    uint modelviewLocation = glGetUniformLocation(shaderProgram, "current_modelview_matrix");
    glUniformMatrix4fvARB(projectionLocation, 1, false, projection);
    glUniformMatrix4fvARB(modelviewLocation, 1, false, modelview);
}

void Shader::sendCameraPosition(Camera* cam)
{
    this->assignShader();
    uint cameraLocation = glGetUniformLocation(shaderProgram, "camera_position");

    glUniform3fARB(cameraLocation, cam->getCameraPosition().x, cam->getCameraPosition().y, cam->getCameraPosition().z);
}


void Shader::createShaderProgram()
{
    shaderProgram = glCreateProgramObjectARB();
    if(vertexShader != 0)
        glAttachObjectARB(shaderProgram,vertexShader);
    if(fragmentShader != 0)
        glAttachObjectARB(shaderProgram,fragmentShader);
    if(geometryShader != 0)
        glAttachObjectARB(shaderProgram,geometryShader);
    glLinkProgramARB(shaderProgram);   
   
    /////Debug////
    PRINT_LOG("__Linking__");
    int logLen;
    char logBuf[1024];
    glGetInfoLogARB(shaderProgram, 1024,&logLen,logBuf);
    PRINT_LOG(logBuf);
}
void Shader::assignShader()
{
    glUseProgramObjectARB(shaderProgram);
}


void Shader::sendMtr4x4Data(const char* name, int count, void* data)
{
        uint unifLocation = glGetUniformLocation(shaderProgram, name);
        glUniformMatrix4fvARB(unifLocation, 1, false, (float*) data);
}