#pragma once
#include <QThread>
#include <QWaitCondition>
#include <QMutex>

class nbodyGLWidget;

class nbodyThread : public QThread {
    Q_OBJECT

    public:
        nbodyThread(nbodyGLWidget* n);

        void wakeup();
        void run();
        void pause();
        void shutdown();

    private:
        bool shouldExit;
        bool iterating;

        nbodyGLWidget* nbodyWidget;
        QWaitCondition renderCond;
        QMutex mutex;

    signals:
        void iterated();
};
