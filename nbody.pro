TEMPLATE = app
CONFIG += qt release mmx sse sse2

TARGET =  nbody
DESTDIR = ./
OBJECTS_DIR = ./obj/

QT += opengl

win32 {
    CONFIG += console
}

DEPENDPATH += src
INCLUDEPATH += src

win32{
    QMAKE_LIBS += -lglew32
}
unix{
    QMAKE_LIBS += -lGLEW -lGLU
}

QMAKE_CXXFLAGS = -msse -msse2 -msse3 -mfpmath=sse -march=native -O3
QMAKE_LFLAGS += -O3

win32 {
    INCLUDEPATH += $(CUDA_INC_DIR)
    QMAKE_LIBDIR += $(CUDA_LIB_DIR)
    LIBS += -lcudart
    
    cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe -c -Xcompiler -fpermissive,$$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
}

unix {
    # auto-detect CUDA path
    CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
    INCLUDEPATH += $$CUDA_DIR/include
    QMAKE_LIBDIR += $$CUDA_DIR/lib
    QMAKE_LIBDIR += $$CUDA_DIR/lib64
    LIBS += -lcudart
    
    cuda.output = ${OBJECTS_DIR}/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = nvcc -c -Xcompiler -fpermissive,$$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
}

cuda.input = CUDA_SOURCES
cuda.name = cuda

QMAKE_EXTRA_COMPILERS += cuda

CUDA_SOURCES = src/nbody/nbody.cu

HEADERS += src/gui/nbody-glwidget.hpp \
           src/gui/nbodythread.hpp \
           src/gui/qtmainwin.hpp \
           src/math/constants.hpp \
           src/math/util.hpp \
           src/math/quaternion.hpp \
           src/nbody/nbody.hpp

SOURCES += src/main.cpp \
           src/gui/nbody-glwidget.cpp \
           src/gui/nbodythread.cpp \
           src/gui/qtmainwin.cpp
