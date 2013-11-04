#include "gpu_utils.h"
#include "cuda_functions.h"


#define CHECK_ERROR_VERBOSE_LEAVE( cuda_error, action_str, leave_label ) \
    if(cuda_error != cudaSuccess) { \
        printf("cuda error at %d : %s \naction: %s \n", __LINE__, cudaGetErrorString(cuda_error), action_str); \
        fflush(stdout); \
        goto leave_label; \
    }

extern "C" unsigned int compare_images(unsigned char* raw_left_image,
                                       unsigned char* raw_right_image,
                                       unsigned short int* diff_result) {
    unsigned char* d_left_pixels;

    unsigned char* d_right_pixels;

    unsigned char* d_diff;

    unsigned char* d_matrix_result;

    unsigned short int* d_vector_result;

    unsigned short* d_result;

    const unsigned int shared_mem_size = sizeof(unsigned short int) * 256;

    dim3 gridDimPixelSub(256, 1);
    dim3 blockDimPixelSub(256, 4);


    dim3 gridDimSumDiffZ(256, 1);
    dim3 blockDimSumDiffZ(256, 1);


    dim3 gridDimSumDiffY(256, 1);
    dim3 blockDimSumDiffY(256, 1);


    dim3 gridDimSumDiffX(1, 1);
    dim3 blockDimSumDiffX(256, 1);


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_left_pixels, TILE_SIZE_BYTES), "alloc left pixels", exit )
    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy(d_left_pixels, raw_left_image, TILE_SIZE_BYTES, cudaMemcpyHostToDevice), "copy left pixels", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_right_pixels, TILE_SIZE_BYTES), "alloc right pixels", exit )
    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy(d_right_pixels, raw_right_image, TILE_SIZE_BYTES, cudaMemcpyHostToDevice), "copy right pixels", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_diff, TILE_SIZE_BYTES), "alloc diff", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_matrix_result, TILE_HEIGHT * TILE_WIDTH * sizeof(unsigned char)), "alloc matrix result", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_vector_result, TILE_WIDTH * sizeof(unsigned short int)), "alloc vector results", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_result, sizeof(unsigned short)), "alloc result", exit )


    sub_one_cube_with_one<<<gridDimPixelSub, blockDimPixelSub>>>(d_left_pixels, d_right_pixels, d_diff);
    CHECK_ERROR_VERBOSE_LEAVE( cudaDeviceSynchronize(), "calc diff", exit )


    sum_z_dimension_one_cude<<<gridDimSumDiffZ, blockDimSumDiffZ>>>(d_diff, d_matrix_result);
    CHECK_ERROR_VERBOSE_LEAVE( cudaDeviceSynchronize(), "reduce by z (R G B A)", exit )


    sum_y_dimension_one_matrix<<<gridDimSumDiffY, blockDimSumDiffY, shared_mem_size>>>(d_matrix_result, d_vector_result);
    CHECK_ERROR_VERBOSE_LEAVE( cudaDeviceSynchronize(), "reduce by y (height)", exit )


    sum_x_dimension_one_vector<<<gridDimSumDiffX, blockDimSumDiffX, shared_mem_size>>>(d_vector_result, d_result);

    CHECK_ERROR_VERBOSE_LEAVE( cudaDeviceSynchronize(), "reduce by x (width)", exit )

    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy(diff_result, d_result, sizeof(unsigned short int), cudaMemcpyDeviceToHost), "copy result", exit )


    exit:

    cudaFree(d_left_pixels);
    cudaFree(d_right_pixels);
    cudaFree(d_diff);
    cudaFree(d_matrix_result);
    cudaFree(d_vector_result);
    cudaFree(d_result);

    return TASK_DONE;
}

unsigned int compare_one_image_with_others(unsigned char* raw_left_image,
                                           unsigned char* raw_right_images,
                                           const unsigned int right_images_count,
                                           unsigned short int* const diff_results) {
    unsigned char* d_left_pixels;


    unsigned char* d_right_pixels;
    size_t right_pixels_pitch;


    unsigned char* d_diff;
    size_t pixels_diff_pitch;


    const size_t results_2d_size = sizeof(unsigned char) * TILE_WIDTH * TILE_HEIGHT;
    unsigned char* d_results_2d;
    size_t results_2d_pitch;


    const size_t results_vector_size = sizeof(unsigned short int) * TILE_WIDTH;
    unsigned short int* d_results_vector;
    size_t resutls_vector_pitch;


    const size_t results_size = sizeof(unsigned short int);
    unsigned short int* d_results;


    const size_t shared_mem_size = sizeof(unsigned short int) * 256;

    dim3 gridDimPixelSub(256, right_images_count); // 256 x 256..1 dep rbs
    dim3 blockDimPixelSub(4, 256);


    dim3 gridDimSumDiffZ(64, right_images_count); // 64 x 256..1 dep rbs
    dim3 blockDimSumDiffZ(4, 256);


    dim3 gridDimSumDiffY(256, right_images_count); // 256 x 256..0 dep rbs
    dim3 blockDimSumDiffY(256, 1);


    dim3 gridDimSumDiffX(right_images_count, 1); // 256..0 x 1 dep rbs
    dim3 blockDimSumDiffX(256, 1);


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_left_pixels, TILE_SIZE_BYTES), "alloc left_pixels", exit )
    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy(d_left_pixels, raw_left_image, TILE_SIZE_BYTES, cudaMemcpyHostToDevice), "copy left_pixels", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMallocPitch((void **)&d_right_pixels, &right_pixels_pitch, TILE_SIZE_BYTES, right_images_count), "alloc right_pixels", exit )
    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy2D((void *)d_right_pixels, right_pixels_pitch, (void *)raw_right_images, TILE_SIZE_BYTES, TILE_SIZE_BYTES, right_images_count, cudaMemcpyHostToDevice), "copy right_pixels", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMallocPitch((void **)&d_diff, &pixels_diff_pitch, TILE_SIZE_BYTES, right_images_count), "allocate diff", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMallocPitch((void **)&d_results_2d, &results_2d_pitch, results_2d_size, right_images_count), "allocate 2d results" , exit)


    CHECK_ERROR_VERBOSE_LEAVE( cudaMallocPitch((void **)&d_results_vector, &resutls_vector_pitch, results_vector_size, right_images_count), "allocate vector results", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_results, results_size * right_images_count), "allocate results", exit )


    sub_one_cube_with_others<<<gridDimPixelSub, blockDimPixelSub>>>(d_left_pixels, d_right_pixels, right_pixels_pitch, d_diff, pixels_diff_pitch);
    CHECK_ERROR_VERBOSE_LEAVE( cudaDeviceSynchronize(), "calc diff", exit )

    sum_z_dimension_zero_or_one<<<gridDimSumDiffZ, blockDimSumDiffZ>>>(d_diff, pixels_diff_pitch, d_results_2d, results_2d_pitch);
    CHECK_ERROR_VERBOSE_LEAVE( cudaDeviceSynchronize(), "reduce by z (R G B A)", exit )

    sum_y_dimension<<<gridDimSumDiffY, blockDimSumDiffY, shared_mem_size>>>(d_results_2d, results_2d_pitch, d_results_vector, resutls_vector_pitch);
    CHECK_ERROR_VERBOSE_LEAVE( cudaDeviceSynchronize(), "reduce by y (height)", exit );

    sum_x_dimension<<<gridDimSumDiffX, blockDimSumDiffX, shared_mem_size>>>(d_results_vector, resutls_vector_pitch, d_results);
    CHECK_ERROR_VERBOSE_LEAVE( cudaDeviceSynchronize(), "reduce by x (width)", exit )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy(diff_results, d_results, results_size * right_images_count, cudaMemcpyDeviceToHost), "copy results", exit )


    exit:

    cudaFree(d_left_pixels);
    cudaFree(d_right_pixels);
    cudaFree(d_diff);
    cudaFree(d_results_2d);
    cudaFree(d_results_vector);
    cudaFree(d_results);


    return TASK_DONE;
}
