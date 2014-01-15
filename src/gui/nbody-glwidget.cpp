#include <GL/glew.h>

#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <cmath>

#include <vector_types.h>

#include "math/quaternion.hpp"
#include "math/constants.hpp"

#include "nbody-glwidget.hpp"
#include "nbodythread.hpp"

using std::cerr;
using std::endl;

nbodyGLWidget::nbodyGLWidget(float3* pos, float3* vel, float* m, const int& n, const int& w, const int& h, QWidget* parent) :
    QGLWidget(QGLFormat(QGL::DoubleBuffer), parent),
    iterating(false),
    positions(pos), velocities(vel), masses(m),
    N(n), _width(w), _height(h), nbodythread(this),
    viewRotX(0.f), viewRotY(0.f), lastPos(0.f, 0.f),
    fovy(45.f), camPos(make_float3(0,0,10)), camLook(make_float3(0,0,0)),
    camForward(normalize(camPos - camLook))
{
    makeCurrent();
    GLenum ret = glewInit();
    if(ret != GLEW_OK) {
        cerr << "Error initializing glew: " << glewGetErrorString(ret) << endl;
        return;
    }else{
        cerr << "Using GLEW version: " << glewGetString(GLEW_VERSION) << endl;
    }

    // If glewInit failed, glGenBuffers will be 0.
    if(glGenBuffers) {
        glGenBuffers(1, &vertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        glBufferData(
                GL_ARRAY_BUFFER,
                N * (3 * sizeof(GLfloat)),
                positions,
                GL_DYNAMIC_DRAW);

        void* posBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(posBuffer, positions, 1*sizeof(float3));
        glUnmapBuffer(GL_ARRAY_BUFFER);
    }

    connect(&nbodythread, SIGNAL(iterated()), this, SLOT(iterated()));
    nbodythread.start();
}

nbodyGLWidget::~nbodyGLWidget() {
    nbodythread.shutdown();
    nbodythread.wait(3000);
}

QSize nbodyGLWidget::minimumSizeHint() const {
    return QSize(_width, _height);
}

QSize nbodyGLWidget::sizeHint() const {
    return QSize(_width, _height);
}

void nbodyGLWidget::initializeGL(){
    enableGLOptions();
}

void nbodyGLWidget::mousePressEvent(QMouseEvent* event) {
    lastPos = event->pos();
    setFocus(Qt::MouseFocusReason);
}

void nbodyGLWidget::mouseMoveEvent(QMouseEvent* event) {
    const float dx = event->x() - lastPos.x();
    const float dy = event->y() - lastPos.y();

    if(dx != 0 || dy != 0){
        viewRotY += sin(TWOPI * (dx / _width));
        viewRotX += sin(TWOPI * (dy / _height));

        if(viewRotY >= TWOPI){
            viewRotY = 0;
        }else if(viewRotY < 0){
            viewRotY = TWOPI;
        }

        if(viewRotX >= HALFPI){
            viewRotX = HALFPI - EPSILON;
        }else if(viewRotX <= -HALFPI){
            viewRotX = -HALFPI + EPSILON;
        }

        lastPos = event->pos();

        update();
    }
}

void nbodyGLWidget::keyPressEvent(QKeyEvent* event){
	// Camera side vector.
	const float3 right = normalize(cross(camForward, make_float3(0, 1, 0)));
    const float3 up = normalize(cross(right, camForward));

    switch(event->key()){
        case Qt::Key_W:
			camPos += camForward / 5.f;
			break;
        case Qt::Key_A:
			camPos -= right / 5.f;
			break;
        case Qt::Key_S:
			camPos -= camForward / 5.f;
			break;
        case Qt::Key_D:
			camPos += right / 5.f;
			break;
        case Qt::Key_Space:
			camPos += up / 5.f;
			break;
        case Qt::Key_Z:
			camPos -= up / 5.f;
			break;
        case Qt::Key_Plus:
        case Qt::Key_Equal:
            fovy += 2;
            break;
        case Qt::Key_Minus:
        case Qt::Key_Underscore:
            fovy -= 2;
            break;
        case Qt::Key_Escape:
            if(iterating) {
                toggleIterating();
            }
            break;
        case Qt::Key_I:
            toggleIterating();
            break;
    }

    update();
}

void nbodyGLWidget::positionCamera() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(fovy, (float)_width / _height, 0.5, 200);

	// Construct quaternions for composite rotation around
	// Y and X axes.
	const quaternion q(cos(-viewRotY/2.0f), make_float3(0, 1, 0) * sin(-viewRotY/2.0f));
	const quaternion r(cos(-viewRotX/2.0f), make_float3(1, 0, 0) * sin(-viewRotX/2.0f));

	// Canonical view vector (quaternion form).
	const quaternion p(0, 0, 0, -1);

	// Rotate p around Y, then around X, then scale by distance.
	const quaternion pr = qmult(qmult(qmult(q, r), p), qmult(q, r).inverse());
	camForward = normalize(make_float3(pr.x, pr.y, pr.z));

    camLook = camPos + camForward;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(camPos.x, camPos.y, camPos.z,
			camLook.x, camLook.y, camLook.z,
			0, 1, 0);
}

void nbodyGLWidget::resizeGL(int w, int h) {
    // Don't allow resizing for now.
    resize(_width, _height);
}

void nbodyGLWidget::paintGL() {
    enableGLOptions();
    positionCamera();

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    // Draw the coordinate axes.
    glBegin(GL_LINES);
        glColor3f(1,0,0);
        glVertex3f(0,0,0);
        glVertex3f(0.25,0,0);

        glColor3f(0,1,0);
        glVertex3f(0,0,0);
        glVertex3f(0,0.25,0);

        glColor3f(0,0,1);
        glVertex3f(0,0,0);
        glVertex3f(0,0,0.25);
    glEnd();

    // Scale the system's units to something more reasonable so the camera
    // position doesn't have to be huge.
    glScalef(1.f/metersPerAU, 1.f/metersPerAU, 1.f/metersPerAU);

    glColor3f(1,0.8,0.5);
    // Use VBOs if they exist, otherwise just draw each point.
    if(glGenBuffers) {
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
        void* posBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
        memcpy(posBuffer, positions, N*sizeof(float3));
        glUnmapBuffer(GL_ARRAY_BUFFER);

        glVertexPointer(3, GL_FLOAT, 0, NULL);

        glDrawArrays(GL_POINTS, 0, N);
    }else {
        glBegin(GL_POINTS);
        for(int i=0; i<N; ++i) {
            // DEBUGGING COLORS
            if(i < N/4)
                glColor3f(0,1,0);
            else if(i < N/3)
                glColor3f(0,0,1);
            else if(i < N/2)
                glColor3f(1,0,0);
            else
                glColor3f(1,1,0);

            // Swap y/z axes since OpenGL's "up" direction is the Y-axis.
            glVertex3f(positions[i].x, positions[i].z, positions[i].y);
        }
        glEnd();
    }

    disableGLOptions();
}

void nbodyGLWidget::enableGLOptions() {
	glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
	glCullFace(GL_BACK);
    glShadeModel(GL_SMOOTH);

	glClearColor(0.0, 0.0, 0.0, 0.0);
    glEnableClientState(GL_VERTEX_ARRAY);
}

void nbodyGLWidget::disableGLOptions() {
	glDisable(GL_MULTISAMPLE);
    glDisableClientState(GL_VERTEX_ARRAY);
}

void nbodyGLWidget::toggleIterating() {
    iterating = !iterating;

    // Wakes up the computing thread so it can go to work.
    if(iterating){
        nbodythread.wakeup();
    }else {
        nbodythread.pause();
    }
}

void nbodyGLWidget::iterated() {
    update();
}
