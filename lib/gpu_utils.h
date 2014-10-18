#ifndef GPU_UTILS_H
#define GPU_UTILS_H

typedef enum TaskStatus {
    TASK_DONE,
    TASK_FAILED
} TaskStatus;

#ifdef __cplusplus

// nvcc compiles as C++ code
#define CUDA_EXPORT extern "C"

#else

// gcc compiles as C code
#define CUDA_EXPORT

#endif


CUDA_EXPORT unsigned int get_max_tiles_count_per_stream();

CUDA_EXPORT TaskStatus compare_one_image_with_others_streams(const unsigned char* const raw_left_image,
                                                             const unsigned char* const raw_right_images,
                                                             const unsigned int right_images_count,
                                                             unsigned int* const diff_results);

CUDA_EXPORT TaskStatus compare_one_image_with_others(const unsigned char* const raw_left_image,
                                                     const unsigned char* const raw_right_images,
                                                     const unsigned int right_images_count,
                                                     unsigned int* const diff_results);

CUDA_EXPORT void* gpu_backend_host_memory_allocator(size_t bytes);
CUDA_EXPORT void gpu_backend_host_memory_deallocator(void* ptr);



#endif // GPU_UTILS_H
