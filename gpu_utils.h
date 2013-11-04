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

extern "C" unsigned int compare_one_image_with_others(unsigned char* raw_left_image,
                                                      unsigned char* raw_right_images,
                                                      const unsigned int right_images_count,
                                                      unsigned short *const diff_results);
#else
// gcc compiles as C code
unsigned int compare_images(unsigned char* raw_left_image,
                            unsigned char* raw_right_image,
                            unsigned short int* diff_result);

unsigned int compare_one_image_with_others(unsigned char* raw_left_image,
                                           unsigned char* raw_right_images,
                                           const unsigned int right_images_count,
                                           unsigned short int* const diff_results);
#endif

#include "tile_utils.h"


#endif // GPU_UTILS_H
