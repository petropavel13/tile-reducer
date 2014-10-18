#include "reduce_test_suite.h"

#include <CUnit/Basic.h>

#include <tile_utils.h>
#include <lodepng.h>
#include <reduce_utils.h>

#include "image_generation_utils.h"

static unsigned char* white_image;
static unsigned char* black_image;
static unsigned char* crossing_image;

static unsigned char* white_image_file;
static size_t white_image_file_size;

static unsigned char* black_image_file;
static size_t black_image_file_size;

static unsigned char* crossing_image_file;
static size_t crossing_image_file_size;


int init_reduce_suite() {
    white_image = generate_white_image(TILE_WIDTH, TILE_HEIGHT);
    black_image = generate_black_image(TILE_WIDTH, TILE_HEIGHT);
    crossing_image = generate_white_black_crossing_squares_image(TILE_WIDTH, TILE_HEIGHT);

    lodepng_encode32(&white_image_file, &white_image_file_size, white_image, TILE_WIDTH, TILE_HEIGHT);
    lodepng_encode32(&black_image_file, &black_image_file_size, black_image, TILE_WIDTH, TILE_HEIGHT);
    lodepng_encode32(&crossing_image_file, &crossing_image_file_size, crossing_image, TILE_WIDTH, TILE_HEIGHT);

    return 0;
}


int clean_reduce_suite() {
    free(white_image);
    free(black_image);
    free(crossing_image);

    free(white_image_file);
    free(black_image_file);
    free(crossing_image_file);

    return 0;
}

void count_equals(reduce_results_t* const node, unsigned int* const equal_count) {
    if (node != NULL) {
        (*equal_count) += node->key == *((unsigned int*)node->data);

        count_equals(node->left, equal_count);
        count_equals(node->right, equal_count);
    }
}

void test_reduce_tiles_single_thread() {
    const unsigned int white_count = 256;
    const unsigned int black_count = 256;
    const unsigned int crossing_count = 256;

    Tile* white_black_tiles[white_count + black_count];
    Tile* white_black_crossing_tiles[white_count + black_count + crossing_count];

    unsigned char* temp_copy = NULL;

    unsigned int tile_idx = 0;

    for (unsigned int i = 0; i < white_count; ++i, ++tile_idx) {
        temp_copy = malloc(white_image_file_size);
        memcpy(temp_copy, white_image_file, white_image_file_size);

        white_black_tiles[i] = tile_new(tile_file_new(temp_copy, white_image_file_size), tile_idx, NULL);


        temp_copy = malloc(white_image_file_size);
        memcpy(temp_copy, white_image_file, white_image_file_size);

        white_black_crossing_tiles[i] = tile_new(tile_file_new(temp_copy, white_image_file_size), tile_idx, NULL);
    }

    for (unsigned int i = white_count; i < white_count + black_count; ++i, ++tile_idx) {
        temp_copy = malloc(black_image_file_size);
        memcpy(temp_copy, black_image_file, black_image_file_size);

        white_black_tiles[i] = tile_new(tile_file_new(temp_copy, black_image_file_size), tile_idx, NULL);


        temp_copy = malloc(black_image_file_size);
        memcpy(temp_copy, black_image_file, black_image_file_size);

        white_black_crossing_tiles[i] = tile_new(tile_file_new(temp_copy, black_image_file_size), tile_idx, NULL);
    }

    for (unsigned int i = white_count + black_count; i < white_count + black_count + crossing_count; ++i, ++tile_idx, NULL) {
        temp_copy = malloc(crossing_image_file_size);
        memcpy(temp_copy, crossing_image_file, crossing_image_file_size);

        white_black_crossing_tiles[i] = tile_new(tile_file_new(temp_copy, crossing_image_file_size), tile_idx, NULL);
    }

    tile_reducer_params params = tile_reducer_params_make_default();

    params.max_cache_size = TILE_SIZE_BYTES * 40;

    reduce_results_t* t_reduce_results = NULL;
    unsigned int t_cnt = 0;

    params.max_num_threads = 1;
    params.max_diff_pixels = 64;
    t_reduce_results = reduce_tiles((Tile* const* const)&white_black_tiles[0], white_count + black_count, params);

    count_equals(t_reduce_results, &t_cnt);

    CU_ASSERT_EQUAL(t_cnt, 2); // reduced to 1 white and 1 black tile
    reduce_results_free(t_reduce_results);


    params.max_diff_pixels = TILE_PIXELS_COUNT;
    t_reduce_results = reduce_tiles((Tile* const* const)&white_black_tiles[0], white_count + black_count, params);

    t_cnt = 0;
    count_equals(t_reduce_results, &t_cnt);

    CU_ASSERT_EQUAL(t_cnt, 1); // reduced to 1 white tile (black tiles absorbed by white)
    reduce_results_free(t_reduce_results);


    params.max_diff_pixels = TILE_PIXELS_COUNT / 2 - 1;
    t_reduce_results = reduce_tiles((Tile* const* const)&white_black_crossing_tiles[0], white_count + black_count + crossing_count, params);

    t_cnt = 0;
    count_equals(t_reduce_results, &t_cnt);

    CU_ASSERT_EQUAL(t_cnt, 3); // reduced to 1 white, 1 black and 1 crossing tile
    reduce_results_free(t_reduce_results);


    params.max_diff_pixels = TILE_PIXELS_COUNT / 2 + 1;
    t_reduce_results = reduce_tiles((Tile* const* const)&white_black_crossing_tiles[0], white_count + black_count + crossing_count, params);

    t_cnt = 0;
    count_equals(t_reduce_results, &t_cnt);

    CU_ASSERT_EQUAL(t_cnt, 2); // reduced to 1 white and 1 black tile (crossing tiles absorbed by white)
    reduce_results_free(t_reduce_results);


    for (unsigned int i = 0; i < white_count; ++i) {
        tile_free(white_black_tiles[i]);

        tile_free(white_black_crossing_tiles[i]);
    }

    for (unsigned int i = white_count; i < white_count + black_count; ++i) {
        tile_free(white_black_tiles[i]);

        tile_free(white_black_crossing_tiles[i]);
    }

    for (unsigned int i = white_count + black_count; i < white_count + black_count + crossing_count; ++i) {
        tile_free(white_black_crossing_tiles[i]);
    }
}
