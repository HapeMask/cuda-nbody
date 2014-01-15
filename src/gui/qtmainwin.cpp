#include <QMenu>
#include <QMenuBar>
#include <QApplication>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QtGui>

#include <vector_types.h>
#include <vector_functions.h>

#include "nbody-glwidget.hpp"
#include "math/constants.hpp"
#include "math/util.hpp"

#include "qtmainwin.hpp"

#include <iostream>
#include <cstdlib>
using namespace std;

nbodyGUI::nbodyGUI(const int& width, const int& height, float3* pos, float3* vel, float* m, const int& n, QWidget* parent) : QMainWindow(parent),
    N(n),
    positions(pos),
    velocities(vel),
    masses(m)
{
    nbodyWidget = new nbodyGLWidget(&positions[0], &velocities[0], &masses[0], N, width, height, parent);
    reload();

    QAction* quit = new QAction("&Quit", this);
    quit->setShortcut(tr("CTRL+Q"));
    QMenu* file = menuBar()->addMenu("&File");
    file->addAction(quit);
    connect(quit, SIGNAL(triggered()), qApp, SLOT(quit()));

    QWidget* win = new QWidget(this);
    QVBoxLayout* vbox = new QVBoxLayout;

    QPushButton* reloadButton = new QPushButton("Reload Simulation", this);
    QPushButton* startStopButton = new QPushButton("Start/Stop Simulation", this);

    connect(startStopButton, SIGNAL(clicked()), nbodyWidget, SLOT(toggleIterating()));
    connect(reloadButton, SIGNAL(clicked()), this, SLOT(reload()));

    vbox->addWidget(nbodyWidget);
    vbox->addWidget(startStopButton);
    vbox->addWidget(reloadButton);

    win->setLayout(vbox);
    setCentralWidget(win);
    nbodyWidget->setFocus(Qt::MouseFocusReason);
}

inline float sampleUniform() {
	return (float)rand() / (float)RAND_MAX;
}

inline float3 sampleHemisphere(const float& u0, const float& u1) {
	const float r = sqrtf(std::max(0.f, 1.f - u0*u0));
	const float phi = TWOPI * u1;
	const float x = r * cos(phi);
	const float y = u0;
	const float z = r * sin(phi);
    return make_float3(x, y, z);
}

inline float3 uniformSampleHemisphere() {
    return sampleHemisphere(sampleUniform(), sampleUniform());
}

inline float3 sampleSphere() {
    // Sample the hemisphere then flip with probability 1/2 to uniformly sample
    // the sphere.
    const float u0 = sampleUniform();
    const float u1 = sampleUniform();

    // Need to rescale u0 to avoid correlation between hemispheres and samples.
    if (u0 < 0.5f) {
        return sampleHemisphere(2.f * u0, u1);
    } else {
        float3 v = sampleHemisphere(2.f * (1 - u0), u1);
        v.y = -v.y;
        return v;
    }
}

// Builds a new random set of positions, masses and velocities.
void nbodyGUI::reload() {
    for(int i=0; i<N; ++i) {
        const float3 pos = sampleSphere()*sampleUniform()*sampleUniform();

        positions[i] = pos * metersPerAU;
        velocities[i] = make_float3(0, 0, 0);

        // Give each body a mass proportional to its distance from the origin
        // (for fun).
        masses[i] = (sampleUniform() - 0.5) * 0.5 * (1.0-norm(pos)) * mEarth;
    }

    nbodyWidget->update();
}
