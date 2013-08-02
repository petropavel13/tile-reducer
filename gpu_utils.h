#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <stdio.h>


#define TASK_DONE 1
#define TASK_FAILED 2


#ifdef __cplusplus
// nvcc compiles as C++ code
extern "C" unsigned int compare_images(unsigned char* raw_left_image,
                           unsigned char* raw_right_image,
                           unsigned short int* diff_result);
#else
// gcc compiles as C code
unsigned int compare_images(unsigned char* raw_left_image,
                           unsigned char* raw_right_image,
                           unsigned short int* diff_result);
#endif

#include "tile_utils.h"


#endif // GPU_UTILS_H
