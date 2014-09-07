#include "cpu_test_suite.h"

#include <CUnit/Basic.h>

#include <tile_utils.h>
#include "tile_utils_i.h"

#include "image_generation_utils.h"

static unsigned char* white_image;
static unsigned char* black_image;
static unsigned char* crossing_image;

int init_cpu_suite(void) {
    white_image = generate_white_image(TILE_WIDTH, TILE_HEIGHT);
    black_image = generate_black_image(TILE_WIDTH, TILE_HEIGHT);
    crossing_image = generate_white_black_crossing_squares_image(TILE_WIDTH, TILE_HEIGHT);

    return 0;
}

int clean_cpu_suite(void) {
    free(white_image);
    free(black_image);
    free(crossing_image);

    return 0;
}

void test_cpu_compare_one_with_one(void) {
    CU_ASSERT_EQUAL(compare_images_one_with_one_cpu(white_image, black_image), TILE_PIXELS_COUNT);
    CU_ASSERT_EQUAL(compare_images_one_with_one_cpu(white_image, crossing_image), TILE_PIXELS_COUNT / 2);

    CU_ASSERT_EQUAL(compare_images_one_with_one_cpu(black_image, white_image), TILE_PIXELS_COUNT);
    CU_ASSERT_EQUAL(compare_images_one_with_one_cpu(black_image, crossing_image), TILE_PIXELS_COUNT / 2);

    CU_ASSERT_EQUAL(compare_images_one_with_one_cpu(crossing_image, white_image), TILE_PIXELS_COUNT / 2);
    CU_ASSERT_EQUAL(compare_images_one_with_one_cpu(crossing_image, black_image), TILE_PIXELS_COUNT / 2);
}

void test_cpu_compare_one_with_many(void) {
    const unsigned int total_count = 1 << 10 /*1024*/;

    unsigned char* const images = malloc(TILE_SIZE_BYTES * total_count);

    for (unsigned int i = 0; i < total_count; i += 2) {
        memcpy(&images[(i + 0) * TILE_SIZE], black_image, TILE_SIZE_BYTES);
        memcpy(&images[(i + 1) * TILE_SIZE], crossing_image, TILE_SIZE_BYTES);
    }

    unsigned int* const results = malloc(sizeof(unsigned int) * total_count);

    compare_images_one_with_many_cpu(white_image, images, total_count, results);

    free(images);

    for (unsigned int i = 0; i < total_count; i += 2) {
        CU_ASSERT_EQUAL(results[i + 0], TILE_PIXELS_COUNT);
        CU_ASSERT_EQUAL(results[i + 1], TILE_PIXELS_COUNT / 2);
    }

    free(results);
}
