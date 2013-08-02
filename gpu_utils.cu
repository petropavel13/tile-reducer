#include "gpu_utils.h"
#include "cuda_functions.h"

extern "C" unsigned int compare_images(unsigned char* raw_left_image,
                                       unsigned char* raw_right_image,
                                       unsigned short int* diff_result) {
#ifdef DEBUG
    printf("call compare_images");
    fflush(stdout);
#endif

    cudaError_t error = cudaSuccess;

    cudaError_t first_error = cudaSuccess;

    unsigned char has_error = 0;

    has_error = 0;
    first_error = cudaSuccess;



    unsigned char* d_left_pixels;

    error = cudaMalloc((void**)&d_left_pixels, TILE_SIZE_BYTES);
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;

    error = cudaMemcpy(d_left_pixels, raw_left_image, TILE_SIZE_BYTES, cudaMemcpyHostToDevice);
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;



    unsigned char* d_right_pixels;

    error = cudaMalloc((void**)&d_right_pixels, TILE_SIZE_BYTES);
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;

    error = cudaMemcpy(d_right_pixels, raw_right_image, TILE_SIZE_BYTES, cudaMemcpyHostToDevice);
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;



    unsigned char* d_diff;

    error = cudaMalloc((void**)&d_diff, TILE_SIZE_BYTES);
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;



    unsigned char* d_matrix_result;

    error = cudaMalloc((void**)&d_matrix_result, TILE_HEIGHT * TILE_WIDTH * sizeof(unsigned char));
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;



    unsigned short int* d_vector_result;

    error = cudaMalloc((void**)&d_vector_result, TILE_WIDTH * sizeof(unsigned short int));
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;



    unsigned short* d_result;

    error = cudaMalloc((void**)&d_result, sizeof(unsigned short));
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;


    const unsigned int shared_mem_size = sizeof(unsigned short int) * 256;

    dim3 gridDimPixelSub(256, 1);
    dim3 blockDimPixelSub(256, 4);

    sub_one_cube_with_one<<<gridDimPixelSub, blockDimPixelSub>>>(d_left_pixels,
                                                                    d_right_pixels,
                                                                    d_diff);

    error = cudaDeviceSynchronize();
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;

    if(has_error) {
        printf("error when calc diff\n");
    }

    // reduce by z (R G B A)

    dim3 gridDimSumDiffZ(256, 1);
    dim3 blockDimSumDiffZ(256, 1);

    sum_z_dimension_one_cude<<<gridDimSumDiffZ, blockDimSumDiffZ>>>(d_diff, d_matrix_result);

    error = cudaDeviceSynchronize();
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;

    if(has_error) {
        printf("error when reduce by z\n");
    }

    // reduce by y (height)

    dim3 gridDimSumDiffY(256, 1);
    dim3 blockDimSumDiffY(256, 1);

    sum_y_dimension_one_matrix<<<gridDimSumDiffY, blockDimSumDiffY, shared_mem_size>>>(d_matrix_result, d_vector_result);

    error = cudaDeviceSynchronize();
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;

    if(has_error) {
        printf("error when reduce by y\n");
    }

    // reduce by x (width)


    dim3 gridDimSumDiffX(1, 1);
    dim3 blockDimSumDiffX(256, 1);

    sum_x_dimension_one_vector<<<gridDimSumDiffX, blockDimSumDiffX, shared_mem_size>>>(d_vector_result, d_result);

    error = cudaDeviceSynchronize();
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;

    if(has_error) {
        printf("error when reduce by x\n");
    }

    unsigned short int result = 42;

    error = cudaMemcpy(&result, d_result, sizeof(unsigned short int), cudaMemcpyDeviceToHost);
    has_error = has_error || error;
    first_error = has_error && !first_error ? error : first_error;

    if(has_error) {
        printf("error when copy results\n");
    }

    (*diff_result) = result;


    cudaFree(d_left_pixels);
    cudaFree(d_right_pixels);
    cudaFree(d_diff);
    cudaFree(d_matrix_result);
    cudaFree(d_vector_result);
    cudaFree(d_result);

    return has_error ? TASK_FAILED : TASK_DONE;
}

