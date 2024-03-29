#include "cuda_functions.h"

__global__ void sub_one_cube_with_others(unsigned char* left_cube,
                                        unsigned char* right_cubes,
                                        size_t right_cube_size_pitch,
                                        unsigned char* diff_cubes,
                                        size_t diff_cube_size_pitch) {

    const unsigned short int x = threadIdx.y; // 0..256
    const unsigned short int z = threadIdx.x; // 0..4
    const unsigned short int y = blockIdx.x; // 0..256

    const unsigned short int cube_number = blockIdx.y; // 0..256


    const unsigned short int x_size = blockDim.y; // 256
    const unsigned short int z_size = blockDim.x; // 4
    // const unsigned short int y_size = gridDim.x; // 256


    const unsigned int index = index_in_3d(x, x_size, y, z, z_size);

    const unsigned int right_offset = right_cube_size_pitch * cube_number;
    const unsigned int diff_offset = diff_cube_size_pitch * cube_number;

    unsigned char* right_cube = (unsigned char*) ((char*)right_cubes + right_offset);
    unsigned char* diff_cube = (unsigned char*) ((char*)diff_cubes + diff_offset);
    

    diff_cube[index] = abs(left_cube[index] - right_cube[index]);
}

__global__ void sum_z_dimension_zero_or_one(unsigned char* cubes,
                                            size_t cube_size_pitch,
                                            unsigned char* matrices_results,
                                            size_t matrix_size_pitch) {

    const unsigned short int y = threadIdx.y; // 0..256
    const unsigned short int x = blockIdx.x * blockDim.x + threadIdx.x; // 0..256 -> 0..64 * 4 + 0..4

    const unsigned short int cube_number = blockIdx.y; // 0..256

    const unsigned short x_size = blockDim.x * gridDim.x; // 4 * 64
    // const unsigned short y_size = blockDim.y; // 256 
    const unsigned short z_size = 4; // 4


    const unsigned int cube_offset = cube_size_pitch * cube_number;
    const unsigned int matrix_offset = matrix_size_pitch * cube_number;

    unsigned char* cube = (unsigned char*) ((char*)cubes + cube_offset);
    unsigned char* matrix = (unsigned char*) ((char*)matrices_results + matrix_offset);

    const unsigned int index_in_matrix = y * x_size + x;

    unsigned short result = 0;

    const unsigned int temp_index = index_in_3d(x, x_size, y, 0, z_size);

    for (unsigned short int i = 0; i < z_size; ++i)
    {
        result += cube[temp_index + i];
    }

    matrix[index_in_matrix] = result > 0; // 0 or 1
}

__global__ void sum_y_dimension(unsigned char* matrices,
                                size_t matrix_size_pitch,
                                unsigned short int* vectors_results,
                                size_t vector_size_pitch) {

    const unsigned short int row_count = blockDim.x; // 256
    const unsigned short int column_count = gridDim.x; // 256

    const unsigned short int matrix_number = blockIdx.y; // 0..256

    const unsigned short int row_number = threadIdx.x; // 0..256
    const unsigned short int column_number = blockIdx.x; // 0..256

    const unsigned int matrix_offset = matrix_size_pitch * matrix_number;
    const unsigned int vector_offset = vector_size_pitch * matrix_number;

    unsigned char* matrix = (unsigned char*) ((char*)matrices + matrix_offset);
    unsigned short int* vector = (unsigned short int*) ((char*)vectors_results + vector_offset);

    const unsigned int row_index_in_matrix = column_count * row_number;

    extern __shared__ unsigned short int temp_results_sum_y[];

    temp_results_sum_y[row_number] = matrix[row_index_in_matrix + column_number];

    __syncthreads();

    for(unsigned short int s = row_count / 2; s >= 1; s = s / 2) // 128, 64, 32, 16, 8, 4, 2, 1
    {
        if(row_number < s)
        {
            temp_results_sum_y[row_number] += temp_results_sum_y[row_number + s];
        }

        __syncthreads();
    }

    if(row_number == 0)
        vector[column_number] = temp_results_sum_y[0];
}

__global__ void sum_x_dimension(unsigned short int* vectors,
                                size_t vector_size_pitch,
                                unsigned int* results) {

    const unsigned short int column_number = threadIdx.x; // 0..256

    const unsigned short int vector_number = blockIdx.x; // 0..256

    const unsigned short int column_count = blockDim.x; // 256

    const unsigned int vector_offset = vector_size_pitch * vector_number;

    unsigned short int* vector = (unsigned short int*) ((char*)vectors + vector_offset);

    extern __shared__ unsigned int temp_results_sum_x[];

    temp_results_sum_x[column_number] = vector[column_number];

    __syncthreads();

    for(unsigned short int s = column_count / 2; s >= 1; s = s / 2) // 128, 64, 32, 16, 8, 4, 2, 1
    {
        if(column_number < s)
        {
            temp_results_sum_x[column_number] += temp_results_sum_x[column_number + s];
        }

        __syncthreads();
    }

    if(column_number == 0)
        results[vector_number] = temp_results_sum_x[0];
}




__global__ void sub_one_cube_with_one(unsigned char* left_cube,
                                        unsigned char* right_cube,
                                        unsigned char* diff_cube) {
    const unsigned short int x = threadIdx.x; // 0..256
    const unsigned short int y = blockIdx.x; // 0..256
    const unsigned short int z = threadIdx.y; // 0..4


    const unsigned int index = index_in_3d(x, 256, y, z, 4);


    diff_cube[index] = abs(left_cube[index] - right_cube[index]);
}


__global__ void sum_z_dimension_one_cude(unsigned char* cube,
                                            unsigned char* matrix_result) {
    const unsigned short int x = threadIdx.x; // 0..256
    const unsigned short int y = blockIdx.x; // 0..256

    const unsigned int index_in_matrix = y * 256 + x;

    unsigned short result = 0;

    const unsigned int temp_index = index_in_3d(x, 256, y, 0, 4);

    for (unsigned short int i = 0; i < 4; ++i)
    {
        result += cube[temp_index + i];
    }

    matrix_result[index_in_matrix] = result > 0; // 0 or 1
}

__global__ void sum_y_dimension_one_matrix(unsigned char* matrix,
                                unsigned short int* vector_result) {
    const unsigned short int row_number = threadIdx.x; // 0..256
    const unsigned short int column_number = blockIdx.x; // 0..256

    const unsigned int row_index_in_matrix = 256 * row_number;

    extern __shared__ unsigned short int temp_results_sum_y[];

    temp_results_sum_y[row_number] = matrix[row_index_in_matrix + column_number];

    __syncthreads();

    for(unsigned short int s = 256 / 2; s >= 1; s = s / 2) // 128, 64, 32, 16, 8, 4, 2, 1
    {
        if(row_number < s)
        {
            temp_results_sum_y[row_number] += temp_results_sum_y[row_number + s];
        }

        __syncthreads();
    }

    if(row_number == 0)
        vector_result[column_number] = temp_results_sum_y[0];
}

__global__ void sum_x_dimension_one_vector(unsigned short int* vector,
                                unsigned short int* result) {
    const unsigned short int column_number = threadIdx.x; // 0..256

    extern __shared__ unsigned int temp_results_sum_x[];

    temp_results_sum_x[column_number] = vector[column_number];

    __syncthreads();

    for(unsigned short int s = 256 / 2; s >= 1; s = s / 2) // 128, 64, 32, 16, 8, 4, 2, 1
    {
        if(column_number < s)
        {
            temp_results_sum_x[column_number] += temp_results_sum_x[column_number + s];
        }

        __syncthreads();
    }

    if(column_number == 0)
        (*result) = temp_results_sum_x[0];
}

