#ifndef __QUATFILE__
#define __QUATFILE__

#include <cmath>

inline float getVecLength(float x, float y, float z)
{
    return sqrt(x * x + y * y + z * z);
}

inline void crossVec(float *veca, float *vecb, float *ret)
{
	ret[0] = veca[1] * vecb[2] - veca[2] * vecb[1];
	ret[1] = veca[2] * vecb[0] - veca[0] * vecb[2];
	ret[2] = veca[0] * vecb[1] - veca[1] * vecb[0];
};

inline float dotVec (float *veca, float *vecb){
	return veca[0] * vecb[0] + veca[1] * vecb[1] + veca[2] * vecb[2];
}

struct Quaternion
{
    float x, y, z, w;
    
    Quaternion():x(0),y(0),z(0),w(0){}
    Quaternion(float x,float y, float z, float angle): x(x), y(y), z(z), w(angle){}
    Quaternion(const vec3& vec, float w):x(vec.x), y(vec.y), z(vec.z), w(w){};
    Quaternion(const Quaternion& cp)
    {
        x = cp.x;
        y = cp.y;
        z = cp.z;
        w = cp.w;
    }
    
    Quaternion operator* (const Quaternion& b)
    {
        float vect1[3]={ x ,y , z};
        float vect2[3]={ b.x, b.y, b.z};
        float retvec[3];
        crossVec(vect1,vect2,retvec);
        Quaternion ret;
        ret.w = w * b.w - dotVec(vect1, vect2);
        ret.x = w * vect2[0] + b.w * vect1[0] + retvec[0];
        ret.y = w * vect2[1] + b.w * vect1[1] + retvec[1];
        ret.z = w * vect2[2] + b.w * vect1[2] + retvec[2];
        return ret;
    }
    Quaternion operator*(const float val)
    {
        return Quaternion(x * val, y * val, z * val, w * val);    
    }

    Quaternion operator+ (const Quaternion& b)
    {
        return Quaternion(x + b.x, y + b.y, z + b.z, w + b.w);
    }

    Quaternion conjugate()
    {
        return Quaternion(-x, -y, -z, w);
    }
    Quaternion inverse()
    {
        return conjugate() * (1.0f / norm());
    }

    float norm()
    {
        return sqrt(x * x + y * y + z * z + w * w);
    }
    void normalize()
    {
        float length = sqrt(x * x + y * y + z * z + w * w);
        x = x / length;
        y = y / length;
        z = z / length;
        w /= length;    
    }
    float getVecNorm()
    {
        return sqrt(x * x + y * y + z * z);
    }
    float getQuatNorm()
    {
        return sqrt(x * x + y * y + z * z + w * w);
    }
    void rotateQuat(vec3 axis, float angle)
    {
        axis.normalize();
        Quaternion axisQuat(axis.x * sin(angle / 2), axis.y * sin(angle / 2),
            axis.z * sin(angle / 2), cos(angle / 2));
        *(this) = axisQuat * (*this);
        *(this) = (*this) * axisQuat.inverse();
    }
    vec3 retVec()
    {
        return vec3(x,y,z);
    }


}; 

#endif