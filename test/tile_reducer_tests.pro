TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.c

HEADERS += \
    tile_utils_i.h

LIBS += -lcunit -lpthread -lpq -lcudart
LIBS += -L../tile_reducer-Debug -ltile_reducer

DEPENDPATH += ../src
INCLUDEPATH += ../src


CUDA_DIR = "/usr/local/cuda-5.5"
CUDA_INCLUDEPATH = $$CUDA_DIR/include

INCLUDEPATH += $$CUDA_INCLUDEPATH

QMAKE_LIBDIR += $$CUDA_DIR/lib64/
