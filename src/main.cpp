#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>

#include <QApplication>

#include <vector_types.h>
#include <vector_functions.h>

#include "nbody/nbody.hpp"
#include "gui/qtmainwin.hpp"

using namespace std;

void integrateRK4(float3* positions, float3* velocities, const float* masses, const float& dt, const int& N);

static const int N = 8192;
int main(int argc, char** args) {
    srand(time(NULL));

    QApplication app(argc, args);

    float3* positions = new float3[N];
    float3* velocities = new float3[N];
    float* masses = new float[N];

    nbodyGUI win(800, 600, positions, velocities, masses, N);

    win.setWindowTitle("CUDA + OpenGL N-Body Simulation");
    win.show();

    int ret = app.exec();
    cleanupCuda();

    return ret;
}
