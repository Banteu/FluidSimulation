#include "structs.h"

vec3::vec3(float x,float y, float z)
{    
        this -> x = x;
        this -> y = y;
        this -> z = z;
}
    
vec3::vec3()
{
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
}
   vec3 vec3::operator+(const vec3& in) const
    {
        return vec3(x + in.x, y + in.y, z + in.z);
    }

   vec3 vec3::operator+(float a) const
    {
        return vec3(x + a, y + a, z + a);
    }
   vec3 vec3::operator*(float in) const
    {
        return vec3(x * in, y * in, z * in);
    }
   vec3::vec3(const vec3& copy)
   {
    this->x = copy.x;
    this->y = copy.y;
    this->z = copy.z;
   }

   float vec3::norm()
   {
    return sqrt(x * x + y * y + z * z);
   }

   void vec3::normalize()
   {
    float len = norm();
    x /= len;
    y /= len;
    z /= len;
   }

  vec3 vec3::operator^(const vec3& in) const
   {
       vec3 returnVec;
        returnVec.x = y * in.z - z * in.y;
	    returnVec.y = z * in.x - x * in.z;
	    returnVec.z = x * in.y - y * in.x;
        return returnVec;
   }

  vec3 vec3::operator-(const vec3& in) const
  {
    vec3 returnVec;
    returnVec.x = x - in.x;
    returnVec.y = y - in.y;
    returnVec.z = z - in.z;
    return returnVec;
  }

  vec3 vec3::operator-(const float in) const
  {
    vec3 returnVec;
    returnVec.x = x - in;
    returnVec.y = y - in;
    returnVec.z = z - in;
    return returnVec;
  }

  vec3 vec3::getAbsVector()
  {
    return vec3(abs(x), abs(y), abs(z));
  }
  vec3& vec3::operator+=(const vec3& in)
  {
	this->x += in.x;
	this->y += in.y;
	this->z += in.z;
	return *this;
  }

 vec3& vec3::operator-=(const vec3& in)
  {
	this->x -= in.x;
	this->y -= in.y;
	this->z -= in.z;
	return *this;
  }

 float vec3::operator*(const vec3& in) const
 {
    return x * in.x + y * in.y + z * in.z;
 }

 vec2 vec3::xy() const
 {
	return vec2(x,y);
 }
 vec2 vec3::xz() const
 {
	return vec2(x,z);
 }
 vec2 vec3::yz() const
 {
	return vec2(y,z);
 }

 /////////////////////////2D Vector //////////////////////////////////////////////////////

 vec2::vec2(float x,float y)
{    
        this -> x = x;
        this -> y = y;
}
    
vec2::vec2()
{
    x = 0.0f;
    y = 0.0f;
}
   vec2 vec2::operator+(const vec2& in) const
    {
        return vec2(x + in.x, y + in.y);
    }
   vec2 vec2::operator*(float in) const
    {
        return vec2(x * in, y * in);
    }
   vec2::vec2(const vec2& copy)
   {
    this->x = copy.x;
    this->y = copy.y;
   }

   float vec2::norm()
   {
    return sqrt(x * x + y * y);
   }

   void vec2::normalize()
   {
    float len = norm();
    x /= len;
    y /= len;
   }

  float vec2::operator^(const vec2& in) const
   {
       return x * in.y - in.x * y;
   }

  vec2 vec2::operator-(const vec2& in) const
  {
    vec2 returnVec;
    returnVec.x = x - in.x;
    returnVec.y = y - in.y;
    return returnVec;
  }

  vec2 vec2::operator-(const float in) const
  {
    vec2 returnVec;
    returnVec.x = x - in;
    returnVec.y = y - in;
    return returnVec;
  }

  vec2 vec2::getAbsVector()
  {
    return vec2(abs(x), abs(y));
  }
  vec2& vec2::operator+=(const vec2& in)
  {
	this->x += in.x;
	this->y += in.y;
	return *this;
  }

 vec2& vec2::operator-=(const vec2& in)
  {
	this->x -= in.x;
	this->y -= in.y;
	return *this;
  }

 float vec2::operator*(const vec2& in) const
 {
    return x * in.x + y * in.y;
 }



 ///////////////////////////////////////// Geometry functions ////////////////////////////

 //// Return value:
 /*   0 - if not laying in triangle
	  1 - if inside triangle
 */

bool pointIsInTriangle(const vec2& point,const vec2& a, const vec2& b, const vec2& c)
 {
	float crab = (b - a) ^ (point - a);
	float crbc = (c - b) ^ (point - b);
	float crca = (a - c) ^ (point - c);

	if(crab * crbc < 0 || crab * crca < 0)
		return 0;

	

		return 1;
 }

 vec3 intresectWithPlane(const vec3& dir, const vec3& point, const vec3& a,
                            const vec3& b, const vec3& c, const vec3& n)
 {


     /* Find intresect with plane */
    float d = -(n * a);
    float t = (n * point - d) / (n * dir);
    return dir * t - point;
    //////////////////////////////////////////

 }


 bool intresectWithTriangle(const vec3& dir, const vec3& point, const vec3& a,
                            const vec3& b, const vec3& c, const vec3& n, vec3& intPoint)
 {

	 /* Get intresection with plane */
	 intPoint = intresectWithPlane(dir,point, a,b,c,n);


	 /* Test othogonality to main planes*/
	 
	 if(n * vec3(1,0,0) != 0)
	 {
		 return pointIsInTriangle(intPoint.yz(), a.yz(), b.yz(), c.yz());	 
	 }
	 if(n * vec3(0,1,0) != 0)
	 {
		 return pointIsInTriangle(intPoint.xz(), a.xz(), b.xz(), c.xz());	 
	 }
	 if(n * vec3(0,0,1) != 0)
	 {
		 return pointIsInTriangle(intPoint.xy(), a.xy(), b.xy(), c.xy());	 
	 }

	 return false;
 }

 vec3 getNormal(const vec3& a,const vec3& b,const vec3& c)
 {
	 return ( (b - a) ^ (c - a) );
 }

 vec4::vec4()
 {
	x = 0;
	y = 0;
	z = 0;
	w = 1;
 }

 vec4::vec4(float x_, float y_, float z_)
 {
	x = x_;
	y = y_;
	z = z_;
	w = 1;
 }

 vec4::vec4(float x_, float y_, float z_, float w_)
 {
	x = x_;
	y = y_;
	z = z_;
	w = w_;
 }

 vec3 vec4::getVec3()
 {
	return vec3( x / w, y / w, z / w); 
 }

 vec4 vec4::operator+(const vec4& b)
 {
	 return vec4( x + b.x, y + b.y, z + b.z, w + b.w);
 }
 
 vec4 vec4::operator-(const vec4& b)
 {
	 return vec4( x - b.x, y - b.y, z - b.z, w - b.w);
 }

  vec4 vec4::operator*(const float vl)
 {
	 return vec4( x * vl, y * vl, z * vl, w * vl);
 }


 vec4::vec4(vec3 in)
 {
	x = in.x;
	y = in.y;
	z = in.z;
	w = 1;
 }