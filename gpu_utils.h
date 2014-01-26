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
#endif

typedef struct DevicePointers {
    unsigned char* left_raw_image;
    unsigned char* right_raw_images;
    size_t right_raw_images_pitch;

    unsigned char* diff_results;
    size_t diff_results_pitch;

    unsigned char* diff_convolution_z;
    size_t diff_convolution_z_pitch;

    unsigned short int* diff_convolution_y;
    size_t diff_convolution_y_pitch;

    unsigned short int* diff_convolution_x;
} DevicePointers;

#include <cuda_runtime.h>

typedef struct RunParams {
    dim3 grid_dim_diff;
    dim3 block_dim_diff;

    dim3 grid_dim_convolution_z;
    dim3 block_dim_convolution_z;

    dim3 grid_dim_convolution_y;
    dim3 block_dim_convolution_y;

    dim3 grid_dim_convolution_x;
    dim3 block_dim_convolution_x;
} RunParams;



#include "tile_utils.h"

cudaError_t alloc_device_mem(DevicePointers* const dps,
                             const unsigned int streams_count,
                             const unsigned int tiles_count);

RunParams make_run_params(const unsigned int stream_size);


cudaError_t run_compare(RunParams rp,
                       DevicePointers dp,
                       cudaStream_t stream,
                       const unsigned int right_images_count,
                       unsigned short int* const diff_results);

#endif // GPU_UTILS_H
