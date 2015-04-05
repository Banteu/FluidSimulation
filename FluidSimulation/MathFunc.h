#pragma once

#ifndef INTERNAL_MATH_FUNC
#define INTERNAL_MATH_FUNC


#include <math.h>
#include <memory>
#include "./structs.h"

/* Matrix 4x4 */ 
void swap_raw_4f(float* vect, int a, int b);
void mult_vect_4f(float* vect, int ind, float mult);
void sub_vect_4f(float* vect, int ind1, int ind2, float mult);

class Matrix4x4f
{
    public:
        Matrix4x4f(float value = 0);
        Matrix4x4f(float* arr);

        
        ~Matrix4x4f();

        Matrix4x4f operator+(const Matrix4x4f& b);
        Matrix4x4f operator-(const Matrix4x4f& b);
        Matrix4x4f operator*(const Matrix4x4f& b);
		vec4	   operator*(const vec4& vector);

        float getValue(int ind) const;
        float getValue(int i, int j) const;
        void setValue(int ind, float value);
        void setValue(int i, int j, float value);
		Matrix4x4f getInversed();
		float* getDataPointer();

        void transpoze();
        Matrix4x4f getTranspozedMatrix();
        void getMainMinor(float* arr);

	private:
        float matrArr[4 * 4];

};





Matrix4x4f getTranslateMatrix(float x, float y, float z);
Matrix4x4f getScaleMatrix(float x, float y, float z);
Matrix4x4f getRotateMatrix(const float angle, const vec3& axis);
Matrix4x4f getIdentityMatrix();

 
#endif