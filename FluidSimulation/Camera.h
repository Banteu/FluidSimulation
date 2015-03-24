#pragma once
#ifndef CAMERA_CLASS_26112013
#define CAMERA_CLASS_26112013
#include "Headers.h"
#include "Quaternion.h"


#define PERSPECTIVE_PROJECTION 1
#define ORTHO_PROJECTION 2


const float DEFAULT_FOV = 70.0f;
const float DEFAULT_ASPECT_RATIO = 1.0;

const float DEFAULTz_NEAR = 0.1;
const float DEFAULTz_FAR = 10;

class Camera
{
public:
    Camera(void);
    Camera(float x,float y, float z, float ax, float ay, float az);
    ~Camera(void);
    void setCameraPos(float x, float y, float z);
    void setAimPos(float x,float y, float z);
    void setUpVec(float x, float y, float z);
    void setRenderMatrix();
    void moveAroundAim(float dx, float dy);
    void rotateAroundAim(float dx, float dy);
    void dollyCam(float df);
    void setAspectRatio(float newAspect);
    void setFov(float newFOV);
    void rotateAroundCamera(const float dx,const float dy);
    vec3 getCameraPosition();
    vec3 getAimPosition();
    void setCameraPos(const vec3& npos);
    void setAimPos(const vec3& npos);
    void setAllPos(float x, float y, float z);
    void setAllPos(const vec3& npos);
    void moveCamera(const vec3& mvec);
    void moveAim(const vec3& mvec);
    
    void setProjectionType(int projectionType);
    void setOrthoBase(uint base);

private:
    vec3 camera;
    vec3 aim;
    vec3 upVect; 
    bool cameraWithAim;
    float fov;
    float aspectRatio;
    float zNear;
    float zFar;

    float orthoBaseSize;

    int PROJECTION_TYPE;




};


#endif
