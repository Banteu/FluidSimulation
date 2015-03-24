#include "Camera.h"

Camera::Camera(void)
{
    cameraWithAim = false;
    camera.x      = 0;
    camera.y      = 0;
    camera.z      = 0;
    aim.x         = 0;
    aim.y         = 0;
    aim.z         = 0;
    upVect.x      = 0;
    upVect.y      = 0;
    upVect.z      = 1;
    
    orthoBaseSize = 100;
    
    fov           = DEFAULT_FOV;
    aspectRatio   = DEFAULT_ASPECT_RATIO;
    zNear         = DEFAULTz_NEAR;
    zFar          = DEFAULTz_FAR;

    PROJECTION_TYPE = PERSPECTIVE_PROJECTION;
}

Camera::Camera(float x,float y,float z, float ax, float ay, float az)
{
    camera.x      = x;
    camera.y      = y;
    camera.z      = z;
    aim.x         = ax;
    aim.y         = ay;
    aim.z         = az;
    
    orthoBaseSize = 100;

    vec3 dir =  aim - camera;
    dir.normalize();

    vec3 locUp = vec3(0, 1, 0);
    if (abs(dir.z) < 0.999)
        locUp = vec3(0, 0, 1);
    vec3 locTang1 = locUp ^ dir;
    upVect =  dir ^ locTang1;

    fov           = DEFAULT_FOV;
    aspectRatio   = DEFAULT_ASPECT_RATIO;
    zNear         = DEFAULTz_NEAR;
    zFar          = DEFAULTz_FAR;
    PROJECTION_TYPE = PERSPECTIVE_PROJECTION;
}


void Camera::rotateAroundAim(float dx, float dy)
{
    vec3 vertAxis(0,0,1);
    vec3 vec = camera - aim;
    Quaternion cameraQuat(vec,0);
    Quaternion upVectQuat(upVect,0);    
    cameraQuat.rotateQuat(vertAxis, dx);    
    upVectQuat.rotateQuat(vertAxis,dx);
    vec3 horizontalAxis = cameraQuat.retVec() ^ upVectQuat.retVec();
    cameraQuat.rotateQuat(horizontalAxis, dy);
    upVectQuat.rotateQuat(horizontalAxis, dy);
    vec = cameraQuat.retVec();
    upVect = horizontalAxis ^ vec;
    upVect.normalize();   
    camera = vec  + aim; 
    return;
}

void Camera::rotateAroundCamera(const float dx,const float dy)
{
    vec3 vertAxis(0,0,1);
    vec3 aimVec = aim - camera;
    Quaternion aimQuat(aimVec, 0);
    Quaternion upVectQuat(vertAxis, 0);
    aimQuat.rotateQuat(vertAxis, dx);    
    upVectQuat.rotateQuat(vertAxis,dx);
    vec3 horizontalAxis = aimQuat.retVec() ^ upVectQuat.retVec();
    aimQuat.rotateQuat(horizontalAxis, dy);
    upVectQuat.rotateQuat(horizontalAxis, dy);
    aimVec = aimQuat.retVec();
    upVect = horizontalAxis ^ aimVec;
    upVect.normalize();   
    aim = aimVec + camera;
    return;
    
}

void Camera::dollyCam(float df)
{
    vec3 vectorCam = camera - aim;
    vectorCam = vectorCam * (df);
    camera = aim + vectorCam;

}
void Camera::setRenderMatrix()
{
    glMatrixMode( GL_PROJECTION ); 
    glLoadIdentity(); 
    if (PROJECTION_TYPE == PERSPECTIVE_PROJECTION)
    {
        gluPerspective(fov,aspectRatio,zNear,zFar);   
    }
    else
    {
        glOrtho(-orthoBaseSize * aspectRatio, orthoBaseSize * aspectRatio, -orthoBaseSize, orthoBaseSize, zNear, zFar);
    }

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    gluLookAt(camera.x,camera.y,camera.z,aim.x,aim.y,aim.z,upVect.x,upVect.y,upVect.z);     
    
    
    return;
}
void Camera::setAspectRatio(float newAscpect)
{
    aspectRatio = newAscpect;
}
void Camera::setFov(float newFOV)
{
    fov = newFOV;
}
Camera::~Camera(void)
{
}


vec3 Camera::getAimPosition()
{
    return aim;
}
vec3 Camera::getCameraPosition()
{
    return camera;
}

void Camera::setCameraPos(float x, float y, float z)
{
    camera = vec3(x,y,z);
}

void Camera::setCameraPos(const vec3& npos )
{
    camera = vec3(npos.x,npos.y,npos.z);
}

void Camera::setAimPos(const vec3& npos )
{
    aim = vec3(npos.x,npos.y,npos.z);
}

void Camera::setAimPos(float x, float y, float z)
{   
    aim = vec3(x,y,z);
}

void Camera::setAllPos(float x, float y, float z)
{   
    setCameraPos(x,y,z);
    setAimPos(x,y,z);
}

void Camera::setAllPos(const vec3& npos)
{   
    setCameraPos(npos);
    setAimPos(npos);
}

void Camera::moveCamera(const vec3& mvec)
{
    camera = camera + mvec;
}

void Camera::moveAim(const vec3& mvec)
{
    aim = aim + mvec;
}

void Camera::setUpVec(float x, float y, float z)
{
    upVect.x = x;
    upVect.y = y;
    upVect.z = z;

}

void Camera::setProjectionType(int prjType)
{
    PROJECTION_TYPE = prjType;
}

void Camera::setOrthoBase(uint base)
{
    orthoBaseSize = base;
}