TEMPLATE = lib
TARGET = treducer
CONFIG += console
CONFIG -= qt
CONFIG -= app_bundle
CONFIG += staticlib

SOURCES += tile_utils.c \
    lodepng.c \
    cache_utils.c \
    generic_avl_tree.c \
    fs_utils.c \
    logging.c \
    reduce_utils.c \
    params.c

OTHER_FILES += \
    cuda_functions.cu \
    gpu_utils.cu \
    cuda_functions.h

HEADERS += \
    lodepng.h \
    tile_utils.h \
    cache_utils.h \
    gpu_utils.h \
    generic_avl_tree.h \
    fs_utils.h \
    logging.h \
    params.h \
    reduce_utils.h

LIBS += -pthread

LIBS += -llog4c
#DEFINES += NO_LOG
#NVCC_OPTIONS += -DNO_LOG

QMAKE_CFLAGS = -std=c99 # for(int i=0)

CUDA_SOURCES += \
    gpu_utils.cu \
    cuda_functions.cu \

CUDA_DIR = "/usr/local/cuda-5.5"

SYSTEM_NAME = unix
SYSTEM_TYPE = 64
CUDA_ARCH = sm_21

NVCC_OPTIONS += --use_fast_math

QMAKE_LIBDIR += $$CUDA_DIR/lib64/

CUDA_OBJECTS_DIR = ./

CUDA_LIBS = -lcudart

CUDA_INCLUDEPATH = $$CUDA_DIR/include

INCLUDEPATH += $$CUDA_INCLUDEPATH

CUDA_INC = $$join(CUDA_INCLUDEPATH,'" -I"','-I"','"')

LIBS += $$CUDA_LIBS


CONFIG(debug, debug|release) {
    # Debug mode
    QMAKE_CXXFLAGS_RELEASE += -Os

    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
# -G generates wrong code! be careful. (cuda 5.5)
#    cuda_d.commands = $$CUDA_DIR/bin/nvcc -G -g -O0 -lineinfo $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -g -O0 -lineinfo $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    QMAKE_CXXFLAGS_RELEASE -= -O2

    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc -O2 $$NVCC_OPTIONS $$CUDA_INC --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
