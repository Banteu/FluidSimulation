#pragma once

#include "Headers.h"
   
extern "C" _declspec(dllexport) void createWind(int argc, char** argv);


extern "C" _declspec(dllexport) void resetSimulation();

extern "C" _declspec(dllexport) void updateRealTimeParams(float viscosity, float dt, float stiffness);

extern "C" _declspec(dllexport) void createPrts(uint count, float volume, float viscosity, float stifness, float density, float dti);