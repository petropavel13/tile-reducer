TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.c \
    image_generation_utils.c \
    cpu_test_suite.c \
    gpu_test_suite.c \
    reduce_test_suite.c

HEADERS += \
    tile_utils_i.h \
    image_generation_utils.h \
    cpu_test_suite.h \
    gpu_test_suite.h \
    reduce_test_suite.h

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
