#include "gpu_utils.h"
#include "cuda_functions.h"

#include <cuda_runtime.h>

#include "math.h"

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

#define DIFF_CONVOLUTION_Z_SIZE (sizeof(unsigned char) * TILE_WIDTH * TILE_HEIGHT)
#define DIFF_CONVOLUTION_Y_SIZE (sizeof(unsigned short int) * TILE_WIDTH)
#define DIFF_CONVOLUTION_X_SIZE (sizeof(unsigned short int))

#define SHARED_MEM_SIZE_WIDTH (sizeof(unsigned short int) * TILE_WIDTH)
#define SHARED_MEM_SIZE_HEIGHT (sizeof(unsigned short int) * TILE_HEIGHT)


#define CHECK_ERROR_VERBOSE_LEAVE( cuda_error, action_str, leave_label, error_var ) \
    if(cuda_error != cudaSuccess) { \
        printf("cuda error at %d : %s \naction: %s \n", __LINE__, cudaGetErrorString(cuda_error), action_str); \
        fflush(stdout); \
        error_var = cuda_error;\
        goto leave_label; \
    }


cudaError_t alloc_device_mem(DevicePointers* const dps,
                             const unsigned int streams_count,
                             const unsigned int tiles_count) {
    cudaError_t error = cudaSuccess;

    for (unsigned int j = 0; j < streams_count; ++j) {

        CHECK_ERROR_VERBOSE_LEAVE( cudaMallocPitch((void**)&dps[j].right_raw_images, &dps[j].right_raw_images_pitch, TILE_SIZE_BYTES, tiles_count), "alloc right_raw_images", exit, error )

        CHECK_ERROR_VERBOSE_LEAVE( cudaMallocPitch((void**)&dps[j].diff_results, &dps[j].diff_results_pitch, TILE_SIZE_BYTES, tiles_count), "alloc diff_results", exit, error )
        CHECK_ERROR_VERBOSE_LEAVE( cudaMallocPitch((void**)&dps[j].diff_convolution_z, &dps[j].diff_convolution_z_pitch, DIFF_CONVOLUTION_Z_SIZE, tiles_count), "alloc diff_convolution_z", exit, error )
        CHECK_ERROR_VERBOSE_LEAVE( cudaMallocPitch((void**)&dps[j].diff_convolution_y, &dps[j].diff_convolution_y_pitch, DIFF_CONVOLUTION_Y_SIZE, tiles_count), "alloc diff_convolution_y", exit, error )
        CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&dps[j].diff_convolution_x, DIFF_CONVOLUTION_X_SIZE * tiles_count), "alloc diff_convolution_x", exit, error )
    }

    exit:

    return error;
}

TaskStatus compare_one_image_with_others_streams(const unsigned char* const raw_left_image,
                                                 const unsigned char* const raw_right_images,
                                                 const unsigned int right_images_count,
                                                 unsigned short int* const diff_results) {

    cudaError_t error = cudaSuccess;

    int cudaDevNumber = 0;
    cudaGetDevice(&cudaDevNumber);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cudaDevNumber);

    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    const unsigned int max_dim_of_grid_of_blocks = deviceProp.major < 3 ? 65535 : (2 << 31) - 1;

    size_t available_memory;
    size_t total_memory;

    cudaMemGetInfo(&available_memory, &total_memory);

    const unsigned int max_r_tiles_by_mem = floor((double)(available_memory - TILE_SIZE_BYTES) / (double)(TILE_SIZE_BYTES + DIFF_CONVOLUTION_Z_SIZE + DIFF_CONVOLUTION_Y_SIZE + DIFF_CONVOLUTION_X_SIZE));
    const unsigned int tiles_per_loop = right_images_count > max_r_tiles_by_mem ? max_r_tiles_by_mem : right_images_count;
    const unsigned int full_loops_count = floor((double)right_images_count / (double)max_r_tiles_by_mem);
    const unsigned int loops_count = ceil((double)right_images_count / (double)max_r_tiles_by_mem);

    const unsigned int max_r_tiles_by_dim = (max_dim_of_grid_of_blocks + 1) / 256;
    const unsigned int tiles_per_stream = tiles_per_loop > max_r_tiles_by_dim ? max_r_tiles_by_dim : tiles_per_loop;
    const unsigned int full_streams_count =  floor((double)tiles_per_loop / (double)max_r_tiles_by_dim);
    const unsigned int streams_count = ceil((double)tiles_per_loop / (double)max_r_tiles_by_dim);


    cudaStream_t streams[streams_count];

    DevicePointers sPointers[streams_count];

    const size_t right_images_size = TILE_SIZE_BYTES * right_images_count;

    unsigned char* pinned_left_raw_image = NULL;
    unsigned char* pinned_right_raw_images = NULL;

    unsigned char* d_left_image = NULL;

    const RunParams rp = make_run_params(tiles_per_stream);

    size_t res_offset = 0;
    size_t tiles_offset = 0;

    unsigned int tail_count = 0;

    CHECK_ERROR_VERBOSE_LEAVE( cudaMallocHost((void**)&pinned_left_raw_image, TILE_SIZE_BYTES), "alloc pinned_left_raw_image ", exit, error )
    CHECK_ERROR_VERBOSE_LEAVE( cudaMallocHost((void**)&pinned_right_raw_images, right_images_size), "alloc pinned_right_raw_image ", exit, error )

    memcpy(pinned_left_raw_image, raw_left_image, TILE_SIZE_BYTES);
    memcpy(pinned_right_raw_images, raw_right_images, right_images_size);


    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc((void**)&d_left_image, TILE_SIZE_BYTES), "alloc left_pixels", exit, error )
    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy((void*)d_left_image, (const void*)raw_left_image, TILE_SIZE_BYTES, H2D), "copy left_pixels", exit, error )

    for (unsigned int j = 0; j < full_streams_count; ++j) {
        sPointers[j].left_raw_image = d_left_image;
        CHECK_ERROR_VERBOSE_LEAVE( cudaStreamCreate(&streams[j]), "create stream", exit, error )
    }

    alloc_device_mem(sPointers, full_streams_count, tiles_per_stream);

    if (streams_count > full_streams_count) {
        tail_count = tiles_per_loop == tiles_per_stream ? tiles_per_loop : tiles_per_loop % tiles_per_stream;

        sPointers[full_streams_count].left_raw_image = d_left_image;

        CHECK_ERROR_VERBOSE_LEAVE( cudaStreamCreate(&streams[full_streams_count]), "create stream", exit, error )

        alloc_device_mem(&sPointers[full_streams_count], 1, tail_count);
    }


    for (unsigned int i = 0; i < full_loops_count; ++i) {
        do_it_again_full:

        res_offset = index_in_3d(0, streams_count, i, 0, tiles_per_stream);
        tiles_offset = index_in_3d(0, streams_count, i, 0, tiles_per_stream * TILE_SIZE_BYTES);

        do {
            for (unsigned int j = 0; j < full_streams_count; ++j) {
                CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy2DAsync((void*)sPointers[j].right_raw_images, sPointers[j].right_raw_images_pitch, (const void*)&pinned_right_raw_images[tiles_offset], TILE_SIZE_BYTES, TILE_SIZE_BYTES, tiles_per_stream, H2D, streams[j]), "copy right_pixels async", inner_error_full, error )

                CHECK_ERROR_VERBOSE_LEAVE( run_compare(rp, sPointers[j], streams[j], tiles_per_stream, &diff_results[res_offset]), "run & copy results", inner_error_full, error )

                res_offset += tiles_per_stream;
                tiles_offset += tiles_per_stream * TILE_SIZE_BYTES;
            }


            if (streams_count > full_streams_count) {
                tail_count = tiles_per_loop % tiles_per_stream;

                RunParams tail_rp = make_run_params(tail_count);

                CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy2DAsync((void*)sPointers[full_streams_count].right_raw_images, sPointers[full_streams_count].right_raw_images_pitch, (const void*)&pinned_right_raw_images[tiles_offset], TILE_SIZE_BYTES, TILE_SIZE_BYTES, tail_count, H2D, streams[full_streams_count]), "copy right_pixels async", inner_error_full, error )

                CHECK_ERROR_VERBOSE_LEAVE( run_compare(tail_rp, sPointers[full_streams_count], streams[full_streams_count], tail_count, &diff_results[res_offset]), "run & copy results", inner_error_full, error )
            }

            inner_error_full:

            if (error != cudaSuccess) {
                cudaDeviceSynchronize();
                cudaDeviceReset();

                goto do_it_again_full;
            } else {
                error = cudaDeviceSynchronize();
            }
        } while (error != cudaSuccess);
    }


    if (loops_count > full_loops_count) {
        const unsigned int tiles_per_tail_loop_count = right_images_count == tiles_per_loop ? right_images_count : right_images_count % tiles_per_loop;

        const unsigned int tail_full_streams_count = floor((double)tiles_per_tail_loop_count / (double)max_r_tiles_by_dim);
        const unsigned int tail_streams_count = ceil((double)tiles_per_tail_loop_count / (double)max_r_tiles_by_dim);

        do {
            for (unsigned int j = 0; j < tail_full_streams_count; ++j) {
                CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy2DAsync((void*)sPointers[j].right_raw_images, sPointers[j].right_raw_images_pitch, (const void*)&pinned_right_raw_images[tiles_offset], TILE_SIZE_BYTES, TILE_SIZE_BYTES, tiles_per_stream, H2D, streams[j]), "copy right_pixels async", inner_error_tail, error )

                CHECK_ERROR_VERBOSE_LEAVE( run_compare(rp, sPointers[j], streams[j], tiles_per_stream, &diff_results[res_offset]), "run & copy results", inner_error_tail, error )

                res_offset += tiles_per_stream;
                tiles_offset += tiles_per_stream * TILE_SIZE_BYTES;
            }


            if (tail_streams_count > tail_full_streams_count) {
                tail_count = tiles_per_tail_loop_count == tiles_per_stream ? tiles_per_tail_loop_count : tiles_per_tail_loop_count % tiles_per_stream;

                RunParams tail_rp = make_run_params(tail_count);
                CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy2DAsync((void*)sPointers[tail_full_streams_count].right_raw_images, sPointers[tail_full_streams_count].right_raw_images_pitch, (const void*)&pinned_right_raw_images[tiles_offset], TILE_SIZE_BYTES, TILE_SIZE_BYTES, tail_count, H2D, streams[tail_full_streams_count]), "copy right_pixels async", inner_error_tail, error )

                CHECK_ERROR_VERBOSE_LEAVE( run_compare(tail_rp, sPointers[tail_full_streams_count], streams[tail_full_streams_count], tail_count, &diff_results[res_offset]), "run & copy results", inner_error_tail, error )
            }

            inner_error_tail:

            if (error != cudaSuccess) {
                cudaDeviceSynchronize();
                cudaDeviceReset();
            } else {
                error = cudaDeviceSynchronize();
            }
        } while (error != cudaSuccess);
    }

    exit:

    for (int j = 0; j < streams_count; ++j) {
       cudaStreamDestroy(streams[j]);
    }

    if (pinned_left_raw_image != NULL) {
        cudaFreeHost(pinned_left_raw_image);
    }

    if (pinned_right_raw_images != NULL) {
        cudaFreeHost(pinned_right_raw_images);
    }


    cudaFree(d_left_image);

    for (unsigned int j = 0; j < streams_count; ++j) {
        cudaFree(sPointers[j].right_raw_images);
        cudaFree(sPointers[j].diff_results);
        cudaFree(sPointers[j].diff_convolution_z);
        cudaFree(sPointers[j].diff_convolution_y);
        cudaFree(sPointers[j].diff_convolution_x);
    }

    return error == cudaSuccess ? TASK_DONE : TASK_FAILED;
}


TaskStatus compare_one_image_with_others(const unsigned char* const raw_left_image,
                                           const unsigned char* const raw_right_images,
                                           const unsigned int right_images_count,
                                           unsigned short int* const diff_results) {
    DevicePointers dp;

    const RunParams rp = make_run_params(right_images_count);

    cudaError_t error = cudaSuccess;

    CHECK_ERROR_VERBOSE_LEAVE( alloc_device_mem(&dp, 1, right_images_count), "alloc device mem", exit, error  )
    CHECK_ERROR_VERBOSE_LEAVE( cudaMalloc(&dp.left_raw_image, TILE_SIZE_BYTES), "alloc left_pixels", exit, error )


    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy((void*)dp.left_raw_image, (const void*)raw_left_image, TILE_SIZE_BYTES, H2D), "copy left_pixels", exit, error )
    CHECK_ERROR_VERBOSE_LEAVE( cudaMemcpy2D((void*)dp.right_raw_images, dp.right_raw_images_pitch, (const void *)raw_right_images, TILE_SIZE_BYTES, TILE_SIZE_BYTES, right_images_count, H2D), "copy right_pixels", exit, error )


    CHECK_ERROR_VERBOSE_LEAVE( run_compare(rp, dp, 0, right_images_count, diff_results), "run & copy results", exit, error )

    error = cudaDeviceSynchronize();

    exit:

    cudaFree(dp.left_raw_image);
    cudaFree(dp.right_raw_images);
    cudaFree(dp.diff_results);
    cudaFree(dp.diff_convolution_z);
    cudaFree(dp.diff_convolution_y);
    cudaFree(dp.diff_convolution_x);


    return error == cudaSuccess ? TASK_DONE : TASK_FAILED;
}

RunParams make_run_params(const unsigned int stream_size) {
    RunParams rp;

    rp.grid_dim_diff = dim3(TILE_WIDTH, stream_size); // 256 x 256..1 dep rbs
    rp.block_dim_diff = dim3(4, TILE_HEIGHT);


    rp.grid_dim_convolution_z = dim3(TILE_WIDTH / 4, stream_size); // 64 x 256..1 dep rbs
    rp.block_dim_convolution_z = dim3(4, TILE_HEIGHT);


    rp.grid_dim_convolution_y = dim3(TILE_WIDTH, stream_size); // 256 x 256..0 dep rbs
    rp.block_dim_convolution_y = dim3(TILE_HEIGHT, 1);


    rp.grid_dim_convolution_x = dim3(stream_size, 1); // 256..0 x 1 dep rbs
    rp.block_dim_convolution_x = dim3(TILE_WIDTH, 1);

    return rp;
}


cudaError_t run_compare(RunParams rp,
                       DevicePointers dp,
                       cudaStream_t stream,
                       const unsigned int right_images_count,
                       unsigned short int* const diff_results) {
    sub_one_cube_with_others<<<rp.grid_dim_diff, rp.block_dim_diff, 0, stream>>>(dp.left_raw_image, dp.right_raw_images, dp.right_raw_images_pitch, dp.diff_results, dp.diff_results_pitch);

    sum_z_dimension_zero_or_one<<<rp.grid_dim_convolution_z, rp.block_dim_convolution_z, 0, stream>>>(dp.diff_results, dp.diff_results_pitch, dp.diff_convolution_z, dp.diff_convolution_z_pitch);

    sum_y_dimension<<<rp.grid_dim_convolution_y, rp.block_dim_convolution_y, SHARED_MEM_SIZE_HEIGHT, stream>>>(dp.diff_convolution_z, dp.diff_convolution_z_pitch, dp.diff_convolution_y, dp.diff_convolution_y_pitch);

    sum_x_dimension<<<rp.grid_dim_convolution_x, rp.block_dim_convolution_x, SHARED_MEM_SIZE_WIDTH, stream>>>(dp.diff_convolution_y, dp.diff_convolution_y_pitch, dp.diff_convolution_x);


    return cudaMemcpyAsync(diff_results, dp.diff_convolution_x, DIFF_CONVOLUTION_X_SIZE * right_images_count, D2H, stream);
}


unsigned int get_max_tiles_count_per_stream() {
    int cudaDevNumber = 0;
    cudaGetDevice(&cudaDevNumber);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cudaDevNumber);

    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    const unsigned int max_dim_of_grid_of_blocks = deviceProp.major < 3 ? 65535 : (2 << 31) - 1;

    size_t available_memory;
    size_t total_memory;

    cudaMemGetInfo(&available_memory, &total_memory);

    const unsigned int max_r_tiles_by_mem = floor((double)(available_memory - TILE_SIZE_BYTES) / (double)(TILE_SIZE_BYTES + DIFF_CONVOLUTION_Z_SIZE + DIFF_CONVOLUTION_Y_SIZE + DIFF_CONVOLUTION_X_SIZE));

    const unsigned int max_r_tiles_by_dim = (max_dim_of_grid_of_blocks + 1) / 256;
    const unsigned int tiles_per_stream = max_r_tiles_by_mem > max_r_tiles_by_dim ? max_r_tiles_by_dim : max_r_tiles_by_mem;

    return tiles_per_stream;
}
