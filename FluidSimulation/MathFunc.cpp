#include "MathFunc.h"


Matrix4x4f::Matrix4x4f(float value)
{
    for(int i = 0; i < 16; ++i)
        matrArr[i] = value;
}
Matrix4x4f::Matrix4x4f(float* arr)
{
    for(int i = 0; i < 16; ++i)
        matrArr[i] = arr[i];
}
Matrix4x4f::~Matrix4x4f()
{
}

float Matrix4x4f::getValue(int ind) const
{
    return matrArr[ind];
}
float Matrix4x4f::getValue(int i, int j) const
{
    return matrArr[4 * i + j];
}

void Matrix4x4f::setValue(int ind, float val)
{
    matrArr[ind] = val;
}
void Matrix4x4f::setValue(int i, int j, float val)
{
    matrArr[4 * i + j] = val;
}

Matrix4x4f Matrix4x4f::operator+(const Matrix4x4f& b)
{
    Matrix4x4f res;
    for(int i = 0; i < 16; ++i)
    {
        res.setValue(i, matrArr[i] + b.matrArr[i]);      
    }
    return res;
}

Matrix4x4f Matrix4x4f::operator-(const Matrix4x4f& b)
{
    Matrix4x4f res;
    for(int i = 0; i < 16; ++i)
    {
        res.setValue(i, matrArr[i] - b.matrArr[i]);      
    }
    return res;
}

Matrix4x4f Matrix4x4f::operator*(const Matrix4x4f& b)
{
    Matrix4x4f res;
    res.matrArr[0] = matrArr[0] * b.matrArr[0] + matrArr[1] * b.matrArr[4] + matrArr[2] * b.matrArr[8] + matrArr[3] * b.matrArr[12];
    res.matrArr[4] = matrArr[4] * b.matrArr[0] + matrArr[5] * b.matrArr[4] + matrArr[6] * b.matrArr[8] + matrArr[7] * b.matrArr[12];
    res.matrArr[8] = matrArr[8] * b.matrArr[0] + matrArr[9] * b.matrArr[4] + matrArr[10] * b.matrArr[8] + matrArr[11] * b.matrArr[12];
    res.matrArr[12] = matrArr[12] * b.matrArr[0] + matrArr[13] * b.matrArr[4] + matrArr[14] * b.matrArr[8] + matrArr[15] * b.matrArr[12];

    res.matrArr[1] = matrArr[0] * b.matrArr[1] + matrArr[1] * b.matrArr[5] + matrArr[2] * b.matrArr[9] + matrArr[3] * b.matrArr[13];
    res.matrArr[5] = matrArr[4] * b.matrArr[1] + matrArr[5] * b.matrArr[5] + matrArr[6] * b.matrArr[9] + matrArr[7] * b.matrArr[13];
    res.matrArr[9] = matrArr[8] * b.matrArr[1] + matrArr[9] * b.matrArr[5] + matrArr[10] * b.matrArr[9] + matrArr[11] * b.matrArr[13];
    res.matrArr[13] = matrArr[12] * b.matrArr[1] + matrArr[13] * b.matrArr[5] + matrArr[14] * b.matrArr[9] + matrArr[15] * b.matrArr[13];

    res.matrArr[2] = matrArr[0] * b.matrArr[2] + matrArr[1] * b.matrArr[6] + matrArr[2] * b.matrArr[10] + matrArr[3] * b.matrArr[14];
    res.matrArr[6] = matrArr[4] * b.matrArr[2] + matrArr[5] * b.matrArr[6] + matrArr[6] * b.matrArr[10] + matrArr[7] * b.matrArr[14];
    res.matrArr[10] = matrArr[8] * b.matrArr[2] + matrArr[9] * b.matrArr[6] + matrArr[10] * b.matrArr[10] + matrArr[11] * b.matrArr[14];
    res.matrArr[14] = matrArr[12] * b.matrArr[2] + matrArr[13] * b.matrArr[6] + matrArr[14] * b.matrArr[10] + matrArr[15] * b.matrArr[14];

    res.matrArr[3] = matrArr[0] * b.matrArr[3] + matrArr[1] * b.matrArr[7] + matrArr[2] * b.matrArr[11] + matrArr[3] * b.matrArr[15];
    res.matrArr[7] = matrArr[4] * b.matrArr[3] + matrArr[5] * b.matrArr[7] + matrArr[6] * b.matrArr[11] + matrArr[7] * b.matrArr[15];
    res.matrArr[11] = matrArr[8] * b.matrArr[3] + matrArr[9] * b.matrArr[7] + matrArr[10] * b.matrArr[11] + matrArr[11] * b.matrArr[15];
    res.matrArr[15] = matrArr[12] * b.matrArr[3] + matrArr[13] * b.matrArr[7] + matrArr[14] * b.matrArr[11] + matrArr[15] * b.matrArr[15];
    return res;
}

Matrix4x4f Matrix4x4f::getInversed()
{
	float temp[16];
	memcpy(temp, matrArr, sizeof(float) * 16);
	float retMatr[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

	
	if(temp[0] == 0)
	{
		if(temp[4] != 0)
		{
			swap_raw_4f(temp,0, 1);
			swap_raw_4f(retMatr,0, 1);
		}
		else
			if(temp[8] != 0)
			{
				swap_raw_4f(temp, 0, 2);
				swap_raw_4f(retMatr,0, 2);
			}
			else
			{
				if(temp[12] != 0)
				{
					swap_raw_4f(temp, 0, 3);
					swap_raw_4f(retMatr,0, 3);
				}
				else return retMatr;
			}
	}

	float val = temp[0];
	mult_vect_4f(temp, 0, 1.0 / val);
	mult_vect_4f(retMatr, 0, 1.0 / val);
	val = temp[4];
	sub_vect_4f(temp, 0, 1, val);
	sub_vect_4f(retMatr, 0, 1, val);
	val = temp[8];
	sub_vect_4f(temp, 0, 2, val);
	sub_vect_4f(retMatr, 0, 2, val);
	val = temp[12];
	sub_vect_4f(temp, 0, 3, val);
	sub_vect_4f(retMatr, 0, 3, val);

	if(temp[5] == 0)
	{
		if(temp[9] != 0)
		{
			swap_raw_4f(temp,1, 2);
			swap_raw_4f(retMatr,1, 2);
		}
		else
			if(temp[13] != 0)
			{
				swap_raw_4f(temp, 1, 3);
				swap_raw_4f(retMatr,1, 3);
			}				
			else return retMatr;
	 }

	val = temp[5];
	mult_vect_4f(temp, 1, 1.0 / val);
	mult_vect_4f(retMatr, 1, 1.0 / val);
	val = temp[9];
	sub_vect_4f(temp, 1, 2, val);
	sub_vect_4f(retMatr, 1, 2, val);
	val = temp[13];
	sub_vect_4f(temp, 1, 3, val);
	sub_vect_4f(retMatr, 1, 3, val);
	
	if(temp[10] == 0)
	{
		if(temp[14] != 0)
		{
			swap_raw_4f(temp,2, 3);
			swap_raw_4f(retMatr,2, 3);
		}
		else return retMatr;
	}

	val = temp[10];
	mult_vect_4f(temp, 2, 1.0 / val);
	mult_vect_4f(retMatr, 2, 1.0 / val);
	val = temp[14];
	sub_vect_4f(temp, 2, 3, val);
	sub_vect_4f(retMatr, 2, 3, val);
	
	if(temp[15] == 0)
		return retMatr;

	val = temp[15];
	mult_vect_4f(temp, 3, 1.0 / val);
	mult_vect_4f(retMatr, 3, 1.0 / val);

	/* Back track*/

	val = temp[11];
	sub_vect_4f(temp, 3, 2, val);
	sub_vect_4f(retMatr, 3, 2, val);
	val = temp[7];
	sub_vect_4f(temp, 3, 1, val);
	sub_vect_4f(retMatr, 3, 1, val);
	val = temp[3];
	sub_vect_4f(temp, 3, 0, val);
	sub_vect_4f(retMatr, 3, 0, val);
	

	val = temp[6];
	sub_vect_4f(temp, 2, 1, val);
	sub_vect_4f(retMatr, 2, 1, val);
	val = temp[2];
	sub_vect_4f(temp, 2, 0, val);
	sub_vect_4f(retMatr, 2, 0, val);
	

	val = temp[1];
	sub_vect_4f(temp, 1, 0, val);
	sub_vect_4f(retMatr, 1, 0, val);
	return retMatr;
}

void swap_raw_4f(float* vect, int a, int b)
{
	std::swap(vect[4 * a], vect[4 * b]);
	std::swap(vect[4 * a + 1], vect[4 * b + 1]);
	std::swap(vect[4 * a + 2], vect[4 * b + 2]);
	std::swap(vect[4 * a + 3], vect[4 * b + 3]);
}
void mult_vect_4f(float* vect, int ind, float mult)
{
	ind *= 4;
	vect[ind++] *= mult;
	vect[ind++] *= mult;
	vect[ind++] *= mult;
	vect[ind] *= mult;
};

void sub_vect_4f(float* vect, int ind1, int ind2, float mult)
{
	ind1 *= 4;
	ind2 *= 4;
	vect[ind2++] -= vect[ind1++] * mult;
	vect[ind2++] -= vect[ind1++] * mult;
	vect[ind2++] -= vect[ind1++] * mult;
	vect[ind2] -= vect[ind1] * mult;
};


vec4 Matrix4x4f::operator*(const vec4& b)
{
	return vec4(matrArr[0] * b.x + matrArr[1] * b.y + matrArr[2] * b.z + matrArr[3] * b.w,
		matrArr[4] * b.x + matrArr[5] * b.y + matrArr[6] * b.z + matrArr[7] * b.w,
		matrArr[8] * b.x + matrArr[9] * b.y + matrArr[10] * b.z + matrArr[11] * b.w,
		matrArr[12] * b.x + matrArr[13] * b.y + matrArr[14] * b.z + matrArr[15] * b.w);
}

float* Matrix4x4f::getDataPointer()
{
	return matrArr;
}

void Matrix4x4f::transpoze()
{
    float temp[16];
    memcpy(temp, matrArr, sizeof(float) * 16);

    matrArr[0] = temp[0];
    matrArr[4] = temp[1];
    matrArr[8] = temp[2];
    matrArr[12] = temp[3];

    matrArr[1] = temp[4];
    matrArr[5] = temp[5];
    matrArr[9] = temp[6];
    matrArr[13] = temp[7];

    matrArr[2] = temp[8];
    matrArr[6] = temp[9];
    matrArr[10] = temp[10];
    matrArr[14] = temp[11];

    matrArr[3] = temp[12];
    matrArr[7] = temp[13];
    matrArr[11] = temp[14];
    matrArr[15] = temp[15];


}


Matrix4x4f getRotateMatrix(const float angle, const vec3& axisGlobal)
{
	vec3 axis = axisGlobal;
	axis.normalize();
	float rotArr[16];

	rotArr[0] = cos(angle) + (1 - cos(angle)) * axis.x * axis.x;
	rotArr[1] = (1 - cos(angle)) * axis.x * axis. y - sin(angle) * axis.z;
	rotArr[2] = (1 - cos(angle)) * axis.x * axis. z + sin(angle) * axis.y;
	rotArr[3] = 0;

	rotArr[4] = (1 - cos(angle)) * axis.x * axis. y + sin(angle) * axis.z;
	rotArr[5] = cos(angle) + (1 - cos(angle)) * axis.y * axis.y;
	rotArr[6] = (1 - cos(angle)) * axis.z * axis. y - sin(angle) * axis.x;
	rotArr[7] = 0;


	rotArr[8] = (1 - cos(angle)) * axis.x * axis. z - sin(angle) * axis.y;	
	rotArr[9] = (1 - cos(angle)) * axis.z * axis. y + sin(angle) * axis.x;
	rotArr[10] = cos(angle) + (1 - cos(angle)) * axis.z * axis.z;
	rotArr[11] = 0;

	rotArr[12] = 0;
	rotArr[13] = 0;
	rotArr[14] = 0;
	rotArr[15] = 1;

	return Matrix4x4f(rotArr);
}

void Matrix4x4f::getMainMinor(float* arr)
{
    arr[0] = matrArr[0];
    arr[1] = matrArr[1];
    arr[2] = matrArr[2];

    arr[3] = matrArr[4];
    arr[4] = matrArr[5];
    arr[5] = matrArr[6];

    arr[6] = matrArr[8];
    arr[7] = matrArr[9];
    arr[8] = matrArr[10];

}

Matrix4x4f getIdentityMatrix()
{
	float arr[] = {1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1};
	return Matrix4x4f(arr);
}


Matrix4x4f getTranslateMatrix(float x, float y, float z)
{
	float arr[16] = {1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1};
	return Matrix4x4f(arr);
}

Matrix4x4f getScaleMatrix(float x, float y, float z)
{
	float arr[16] = {x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1};
	return Matrix4x4f(arr);
}