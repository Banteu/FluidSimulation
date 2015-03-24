#ifndef BASE_STRUCTS
#define BASE_STRUCTS
#include <math.h>



typedef unsigned int uint;


struct vec2
{
    float x,y;
    vec2(float x,float y);
    vec2();
    vec2(const vec2& copy);
    vec2 operator+(const vec2& in)  const;
    
    vec2 operator*(float in)        const;
    float operator^ (const vec2& in) const;
    vec2 operator-(const vec2& in)  const;
    vec2 operator-(const float in)  const;
	vec2& operator+=(const vec2& in);
	vec2& operator-=(const vec2& in);
    float operator*(const vec2& in) const;
    vec2 getAbsVector();
    float norm();
    void normalize();   
};

struct vec3
{
    float x,y,z;
    vec3(float x,float y,float z);
    vec3();
    vec3(const vec3& copy);
    vec3 operator+(const vec3& in)  const;
    vec3 operator+(float a) const;
    vec3 operator*(float in)        const;
    vec3 operator^ (const vec3& in) const;
    vec3 operator-(const vec3& in)  const;
    vec3 operator-(const float in)  const;
	vec3& operator+=(const vec3& in);
	vec3& operator-=(const vec3& in);
    float operator*(const vec3& in) const;
    vec3 getAbsVector();

	vec2 xy() const;
	vec2 xz() const ;
	vec2 yz() const;

    float norm();
    void normalize();   
};


struct vec4
{
	float x,y,z,w;
	vec4();
	vec4(vec3);
	vec4(float x, float y, float z);
    vec4(float x, float y, float z, float w);
	

	vec3 getVec3();

	vec4 operator+(const vec4& b);
	vec4 operator-(const vec4& b);


};

bool pointIsInTriangle(const vec2& point,const vec2& a, const vec2& b, const vec2& c);

bool intresectWithTriangle(const vec3& dir, const vec3& point, const vec3& a,
                            const vec3& b, const vec3& c, const vec3& n, vec3& intPoint);
vec3 intresectWithPlane(const vec3& line, const vec3& pointOnLine,  const vec3& a, const vec3& b, const vec3& c, const vec3& norm);

vec3 getNormal(const vec3& a, const vec3& b, const vec3& c);

#endif