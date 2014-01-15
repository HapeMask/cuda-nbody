#pragma once

#include <QMainWindow>
#include <QLabel>

#include "nbody-glwidget.hpp"

class nbodyGUI : public QMainWindow {
    Q_OBJECT

    public:
        nbodyGUI(const int& width, const int& height, float3* pos, float3* vel, float* m, const int& n, QWidget* parent = NULL);

    private:
        nbodyGLWidget* nbodyWidget;

        int N;
        float3* positions;
        float3* velocities;
        float* masses;

    public slots:
        void reload();
};
