#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <stdio.h>


typedef enum TaskStatus {
    TASK_DONE,
    TASK_FAILED
} TaskStatus;


#ifdef __cplusplus
// nvcc compiles as C++ code

extern "C" unsigned int get_max_tiles_count_per_stream();

extern "C" TaskStatus compare_one_image_with_others_streams(const unsigned char * const raw_left_image,
                                                 const unsigned char * const raw_right_images,
                                                 const unsigned int right_images_count,
                                                 unsigned short int* const diff_results);

extern "C" TaskStatus compare_one_image_with_others(const unsigned char * const raw_left_image,
                                                      const unsigned char * const raw_right_images,
                                                      const unsigned int right_images_count,
                                                      unsigned short *const diff_results);

extern "C" void* gpu_backend_memory_allocator(size_t bytes);
extern "C" void gpu_backend_memory_deallocator(void* ptr);
#else
// gcc compiles as C code

unsigned int get_max_tiles_count_per_stream();

TaskStatus compare_one_image_with_others_streams(const unsigned char * const raw_left_image,
                                                 const unsigned char * const raw_right_images,
                                                 const unsigned int right_images_count,
                                                 unsigned short int* const diff_results);

TaskStatus compare_one_image_with_others(const unsigned char * const raw_left_image,
                                                      const unsigned char * const raw_right_images,
                                                      const unsigned int right_images_count,
                                                      unsigned short *const diff_results);

void* gpu_backend_memory_allocator(size_t bytes);
void gpu_backend_memory_deallocator(void* ptr);
#endif


#endif // GPU_UTILS_H
