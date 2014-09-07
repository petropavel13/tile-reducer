#ifndef GPU_TEST_SUITE_H
#define GPU_TEST_SUITE_H

int init_gpu_suite(void);
int clean_gpu_suite(void);

void test_gpu_compare_one_to_many(void);
void test_gpu_compare_one_to_many_streams(void);

#endif // GPU_TEST_SUITE_H
