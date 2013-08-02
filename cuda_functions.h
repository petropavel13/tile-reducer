#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

__device__ inline unsigned int index_in_3d(unsigned int x,
                                        unsigned int x_size,
                                        unsigned int y,
                                        unsigned int z,
                                        unsigned int z_size) {
    return y * x_size * z_size + x * z_size + z;
}

extern __host__ __device__ int abs(int) __THROW;


__global__ void sub_one_cube_with_others(unsigned short int left_offset_start,
                                        unsigned char* left_cubes,
                                        size_t left_cube_size_pitch,
                                        unsigned char* right_cubes,
                                        size_t right_cube_size_pitch,
                                        unsigned char* diff_cubes,
                                        size_t diff_cube_size_pitch);


__global__ void sum_z_dimension_zero_or_one(unsigned char* cubes,
                                            size_t cube_size_pitch,
                                            unsigned char* matrices_results,
                                            size_t matrix_size_pitch);


__global__ void sum_y_dimension(unsigned char* matrices,
                                size_t matrix_size_pitch,
                                unsigned short int* vectors_results,
                                size_t vector_size_pitch);


__global__ void sum_x_dimension(unsigned short int* vectors,
                                size_t vector_size_pitch,
                                unsigned short int* results);





__global__ void sub_one_cube_with_one(unsigned char* left_cube,
                                        unsigned char* right_cube,
                                        unsigned char* diff_cube);


__global__ void sum_z_dimension_one_cude(unsigned char* cube,
                                            unsigned char* matrix_result);


__global__ void sum_y_dimension_one_matrix(unsigned char* matrix,
                                unsigned short int* vector_result);


__global__ void sum_x_dimension_one_vector(unsigned short int* vector,
                                unsigned short int* result);

#endif // CUDA_FUNCTIONS_H
