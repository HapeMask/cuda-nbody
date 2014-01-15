#pragma once
#include <QGLWidget>
#include <QPaintEvent>
#include <QKeyEvent>

#include <vector_types.h>

#include "nbodythread.hpp"

class nbodyGLWidget : public QGLWidget {
    Q_OBJECT

    friend class nbodyThread;

    public:
        nbodyGLWidget(float3* pos, float3* vel, float* m, const int& n, const int& w, const int& h, QWidget* parent = NULL);
        virtual ~nbodyGLWidget();

        QSize minimumSizeHint() const;
        QSize sizeHint() const;

    protected:
        void initializeGL();
        void resizeGL(int width, int height);

        void paintGL();
        void mousePressEvent(QMouseEvent* event);
        void mouseMoveEvent(QMouseEvent* event);
        void keyPressEvent(QKeyEvent* event);

    private:
		bool iterating;

        void enableGLOptions();
        void disableGLOptions();
        void positionCamera();

        float3* positions;
        float3* velocities;
        float* masses;
        int N;

        int _width, _height;
        nbodyThread nbodythread;

        float viewRotX, viewRotY;
        QPoint lastPos;
        float fovy;
        float3 camPos, camLook, camForward;

        GLuint vertexBuffer;

    private slots:
        void toggleIterating();
        void iterated();
};
