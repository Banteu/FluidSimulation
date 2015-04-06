#pragma once

#ifndef _MAIN_HEADERS
#define _MAIN_HEADERS

#include <Windows.h>
#include <GL\glew.h>
#include <GL\freeglut.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include "structs.h"
#include <thrust\sort.h>
#include <thrust\device_ptr.h>
#include "DLLInterface.h"
#include <vector>

class Camera;
class Shader;

#include "Material.h"
#include "Camera.h"
#include "SphContainer.h"
#include "SphCpuSolver.h"
#include "MathFunc.h"
#include <string>
bool createWindow(int argc, char **argv);
#endif