TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.c

QMAKE_CFLAGS = -std=c99

LIBS += -lcunit -lpthread -lpq -lcudart -llog4c
LIBS += -L../../build/tile_reducer_lib-Release -ltreducer
LIBS += -L../../build/tile_reducer_lib-Debug -ltreducer

DEPENDPATH += ../lib
INCLUDEPATH += ../lib

CUDA_DIR = "/usr/local/cuda-5.5"
CUDA_INCLUDEPATH = $$CUDA_DIR/include

INCLUDEPATH += $$CUDA_INCLUDEPATH

QMAKE_LIBDIR += $$CUDA_DIR/lib64/
