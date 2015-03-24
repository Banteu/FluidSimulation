#include "Headers.h"
#include "DLLInterface.h"

extern "C" _declspec(dllexport) void createWind(int argc, char** argv)
{
    createWindow(argc, argv);
}




