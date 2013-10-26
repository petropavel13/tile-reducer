TEMPLATE = app
CONFIG += console
CONFIG -= qt
CONFIG -= app_bundle

SOURCES += main.c \
    tile_utils.c \
    lodepng.c \
    cache_utils.c \
    db_utils.c \
    cluster_utils.c \
    generic_avl_tree.c \
    color_index_utils.c

OTHER_FILES += \
    cuda_functions.cu \
    gpu_utils.cu \
    cuda_functions.h

HEADERS += \
    lodepng.h \
    tile_utils.h \
    cache_utils.h \
    db_utils.h \
    gpu_utils.h \
    cluster_utils.h \
    generic_avl_tree.h \
    color_index_utils.h

INCLUDEPATH += "/usr/include/postgresql"

LIBS += -lpq

QMAKE_CFLAGS = -std=c99 -D_BSD_SOURCE # for(int i=0), DT_DIR

CUDA_SOURCES += \
    gpu_utils.cu \
    cuda_functions.cu

CUDA_DIR = "/usr/local/cuda-5.5"

SYSTEM_NAME = unix
SYSTEM_TYPE = 64
CUDA_ARCH = sm_21

NVCC_OPTIONS = --use_fast_math

QMAKE_LIBDIR += $$CUDA_DIR/lib64/

#QMAKE_LIBDIR += $$CUDA_DIR/lib64/ \
    #"/usr/lib64/nvidia-current/"

CUDA_OBJECTS_DIR = ./

CUDA_LIBS = -lcudart
#CUDA_LIBS = -lcuda -lcudart

CUDA_INCLUDEPATH = $$CUDA_DIR/include

INCLUDEPATH += $$CUDA_INCLUDEPATH

CUDA_INC = $$join(CUDA_INCLUDEPATH,'" -I"','-I"','"')

LIBS += $$CUDA_LIBS


CONFIG(debug, debug|release) {
    # Debug mode
    QMAKE_CXXFLAGS_RELEASE += -Os

    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
#    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -G -g -O0 -lineinfo $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    QMAKE_CXXFLAGS_RELEASE -= -O2

    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
#    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}.o
    cuda.commands = $$CUDA_DIR/bin/nvcc -O2 $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
