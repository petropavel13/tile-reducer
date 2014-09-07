#include <stdio.h>
#include <CUnit/Basic.h>
#include "../src/tile_utils.h"
#include "tile_utils_i.h"

#include <stdlib.h>

int init_suite1(void) {
    // TODO: generate images
    return 0;
}

int clean_suite1(void) {
    // TODO: delete generated images
    return 0;
}

void test_tiles_equals_one_to_one(void) {
    char* path0 = realpath("../test/images/equals/524.png", NULL);
    char* path1 = realpath("../test/images/equals/525.png", NULL);

    CU_ASSERT_PTR_NOT_NULL_FATAL(path0);
    TileFile* const tf0 = read_tile(path0);
    free(path0);

    CU_ASSERT_PTR_NOT_NULL_FATAL(path1);
    TileFile* const tf1 = read_tile(path1);
    free(path1);

    unsigned char* t0pixels = NULL;
    unsigned char* t1pixels = NULL;

    CU_ASSERT_EQUAL(get_tile_pixels(tf0, &t0pixels), 0);
    CU_ASSERT_EQUAL(get_tile_pixels(tf1, &t1pixels), 0);

    tile_file_destructor(tf0);
    tile_file_destructor(tf1);

    const unsigned int diff_pixels_count = compare_images_one_with_one_cpu(t0pixels, t1pixels);

    free(t0pixels);
    free(t1pixels);

    CU_ASSERT_EQUAL(diff_pixels_count, 0);
}

void test_tiles_not_equals_one_to_one(void) {
    char* path0 = realpath("../test/images/not_equals/312.png", NULL);
    char* path1 = realpath("../test/images/not_equals/313.png", NULL);

    CU_ASSERT_PTR_NOT_NULL_FATAL(path0);
    TileFile* const tf0 = read_tile(path0);
    free(path0);

    CU_ASSERT_PTR_NOT_NULL_FATAL(path1);
    TileFile* const tf1 = read_tile(path1);
    free(path1);

    unsigned char* t0pixels = NULL;
    unsigned char* t1pixels = NULL;

    CU_ASSERT_EQUAL(get_tile_pixels(tf0, &t0pixels), 0);
    CU_ASSERT_EQUAL(get_tile_pixels(tf1, &t1pixels), 0);

    tile_file_destructor(tf0);
    tile_file_destructor(tf1);

    const unsigned int diff_pixels_count = compare_images_one_with_one_cpu(t0pixels, t1pixels);

    free(t0pixels);
    free(t1pixels);

    CU_ASSERT_NOT_EQUAL(diff_pixels_count, 0);
}



int main(void)
{
    CU_pSuite pSuite = NULL;

   /* initialize the CUnit test registry */
   if (CUE_SUCCESS != CU_initialize_registry())
      return CU_get_error();

   /* add a suite to the registry */
   pSuite = CU_add_suite("Suite_1", init_suite1, clean_suite1);
   if (pSuite == NULL) {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* add the tests to the suite */
   if (CU_add_test(pSuite, "test of tiles equals (one-to-one)", test_tiles_equals_one_to_one) == NULL
           || CU_add_test(pSuite, "tiles not equals (one-to-one)", test_tiles_not_equals_one_to_one) == NULL
//           || CU_add_test(pSuite, "tiles not equals (one-to-one)", test_tiles_not_equals_one_to_one) == NULL
       )
   {
      CU_cleanup_registry();
      return CU_get_error();
   }

   /* Run all tests using the CUnit Basic interface */
   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();
   CU_cleanup_registry();
   return CU_get_error();
}

