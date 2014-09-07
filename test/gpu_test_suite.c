#include "gpu_test_suite.h"

#include <CUnit/Basic.h>

#include "image_generation_utils.h"

#include <tile_utils.h>
#include <gpu_utils.h>

static unsigned char* white_image;
static unsigned char* black_image;
static unsigned char* crossing_image;


int init_gpu_suite(void) {
    white_image = generate_white_image(TILE_WIDTH, TILE_HEIGHT);
    black_image = generate_black_image(TILE_WIDTH, TILE_HEIGHT);
    crossing_image = generate_white_black_crossing_squares_image(TILE_WIDTH, TILE_HEIGHT);

    return 0;
}

int clean_gpu_suite(void) {
    free(white_image);
    free(black_image);
    free(crossing_image);

    return 0;
}

void test_gpu_compare_one_to_many(void) {
    const unsigned int total_count = 1 << 10 /*1024*/;

    unsigned char* const images = gpu_backend_host_memory_allocator(TILE_SIZE_BYTES * total_count);

    for (unsigned int i = 0; i < total_count; i += 2) {
        memcpy(&images[(i + 0) * TILE_SIZE], black_image, TILE_SIZE_BYTES);
        memcpy(&images[(i + 1) * TILE_SIZE], crossing_image, TILE_SIZE_BYTES);
    }

    unsigned int* const results = malloc(sizeof(unsigned int) * total_count);

    CU_ASSERT_EQUAL(compare_one_image_with_others(white_image, images, total_count, results), TASK_DONE);

    gpu_backend_host_memory_deallocator(images);

    for (unsigned int i = 0; i < total_count; i += 2) {
        CU_ASSERT_EQUAL(results[i + 0], TILE_PIXELS_COUNT);
        CU_ASSERT_EQUAL(results[i + 1], TILE_PIXELS_COUNT / 2);
    }

    free(results);
}

void test_gpu_compare_one_to_many_streams(void) {
    const unsigned int total_count = get_max_tiles_count_per_stream() * 3 + 16; // ~3 streams

    unsigned char* const images = gpu_backend_host_memory_allocator(TILE_SIZE_BYTES * total_count);

    for (unsigned int i = 0; i < total_count; i += 2) {
        memcpy(&images[(i + 0) * TILE_SIZE], black_image, TILE_SIZE_BYTES);
        memcpy(&images[(i + 1) * TILE_SIZE], crossing_image, TILE_SIZE_BYTES);
    }

    unsigned int* const results = malloc(sizeof(unsigned int) * total_count);

    CU_ASSERT_EQUAL(compare_one_image_with_others_streams(white_image, images, total_count, results), TASK_DONE);

    gpu_backend_host_memory_deallocator(images);

    for (unsigned int i = 0; i < total_count; i += 2) {
        CU_ASSERT_EQUAL(results[i + 0], TILE_PIXELS_COUNT);
        CU_ASSERT_EQUAL(results[i + 1], TILE_PIXELS_COUNT / 2);
    }

    free(results);
}
