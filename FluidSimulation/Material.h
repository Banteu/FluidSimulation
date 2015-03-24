#pragma once
#ifndef MATERIAL_CLASS_17022014
#define MATERIAL_CLASS_17022014

#include "Headers.h"

// In all shaders use current_projection_matrix and current_modelview_matrix

using std::string;

extern FILE* LOG_FILE_POINTER;

inline void PRINT_LOG(string str)
{
    if(LOG_FILE_POINTER == 0)
        return;

    fprintf(LOG_FILE_POINTER, str.c_str());
    fprintf(LOG_FILE_POINTER, "\n");
    fflush(LOG_FILE_POINTER);
}



class Shader
{
public:

    Shader();
    ~Shader();
    void createShader(const string& vertexShader, const string& fragmentShader, const string& geometryShader);
    void assignShader();
    void getCode(const string& filename, char** saveTo, int& count);
    void readVertexShader(const string& filename);
    void readFragmentShader(const string& filename);
    void readGeometryShader(const string& filename);
    void sendDataBlock(int location, void* data);
    void sendArrayData(int location,int count, void* data);
    void sendMtr4x4Data(const char* name, int count, void* data);
    void sendViewMatrices(float* projection, float* modelview);
    void sendFloat(const char* name, float value);
    void sendInt(const char* name, int value);
    void sendCameraPosition(Camera* cam);

    uint getProgram();
    void createShaderProgram();
private:
    char* vertexShaderCodeBuffer;
    int  vertexShaderSize;

    char* fragmentShaderCodeBuffer;
    int  fragmentShaderSize;

    char* geometryShaderCodeBuffer;
    int  geometryShaderSize;

    uint vertexShader;
    uint fragmentShader;
    uint geometryShader;
    uint shaderProgram;
};


#endif
