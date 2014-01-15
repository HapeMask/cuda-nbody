#include <sys/time.h>

#include "nbodythread.hpp"
#include "nbody-glwidget.hpp"

#include "nbody/nbody.hpp"

nbodyThread::nbodyThread(nbodyGLWidget* n) : shouldExit(false), iterating(false), nbodyWidget(n) {}
static const double dt = 365 * (60 * 60 * 24);

#include <iostream>
using namespace std;
void nbodyThread::run() {
    timeval start, end;

    while(!shouldExit) {
        mutex.lock();
        if(!iterating){
            renderCond.wait(&mutex);
        }
        mutex.unlock();

        gettimeofday(&start, NULL);
        integrateRK4(
                nbodyWidget->positions,
                nbodyWidget->velocities,
                nbodyWidget->masses,
                dt, nbodyWidget->N);
        gettimeofday(&end, NULL);

        const float msElapsed = (end.tv_sec - start.tv_sec) * 1000.f + (end.tv_usec - start.tv_usec) / 1000.f;
        if(msElapsed < 33) {
            msleep(33 - msElapsed);
        }

        emit iterated();
    }
}

void nbodyThread::wakeup() {
    mutex.lock();
    iterating = true;
    mutex.unlock();

    renderCond.wakeAll();
}

void nbodyThread::pause() {
    mutex.lock();
    iterating = false;
    mutex.unlock();
}

void nbodyThread::shutdown() {
    mutex.lock();
    iterating = false;
    shouldExit = true;
    mutex.unlock();

    renderCond.wakeAll();
}
