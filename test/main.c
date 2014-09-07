#include <CUnit/Basic.h>

#include "cpu_test_suite.h"
#include "gpu_test_suite.h"

int main(void)
{
    if (CU_initialize_registry() != CUE_SUCCESS)
       return CU_get_error();


   const CU_pSuite cpu_suite = CU_add_suite("CPU", init_cpu_suite, clean_cpu_suite);

   if (cpu_suite == NULL) goto exit;

   if (CU_add_test(cpu_suite, "compare one to one", test_cpu_compare_one_with_one) == NULL
           || CU_add_test(cpu_suite, "compare one to many", test_cpu_compare_one_with_many) == NULL
       ) {
      goto exit;
   }

   const CU_pSuite gpu_suite = CU_add_suite("CUDA", init_gpu_suite, clean_gpu_suite);

   if (gpu_suite == NULL) goto exit;

   if (CU_add_test(gpu_suite, "compare one to many", test_gpu_compare_one_to_many) == NULL
           || CU_add_test(gpu_suite, "compare one to many on multiple streams", test_gpu_compare_one_to_many_streams) == NULL
      ) {
       goto exit;
   }

   CU_basic_set_mode(CU_BRM_VERBOSE);
   CU_basic_run_tests();

   exit:

   CU_cleanup_registry();
   return CU_get_error();
}

