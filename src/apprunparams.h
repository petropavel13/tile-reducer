#ifndef RUNPARAMS_H
#define RUNPARAMS_H

#include "stddef.h" // for size_t

typedef struct AppRunParams {
    unsigned short int max_diff_pixels;
    unsigned char max_num_threads;
    size_t max_cache_size;
    unsigned int min_tiles_count_for_gpu_compare;
    size_t max_sql_string_size;
} AppRunParams;


static inline AppRunParams make_default_app_run_params() {
    return (AppRunParams){
        .max_diff_pixels = 32,
        .max_num_threads = 2,
        .max_cache_size = 512 * 1024 * 1024,
        .min_tiles_count_for_gpu_compare = 128,
        .max_sql_string_size = 8 * 1024 * 1024,
    };
}

#endif // RUNPARAMS_H
