#ifndef RUNPARAMS_H
#define RUNPARAMS_H

#include "stddef.h" // for size_t

typedef struct tile_reducer_params {
    unsigned int max_diff_pixels;
    unsigned char max_num_threads;
    size_t max_cache_size;
    unsigned int min_tiles_count_for_gpu_compare;
    char path[256];
} tile_reducer_params;


static inline tile_reducer_params tile_reducer_params_make_default() {
    tile_reducer_params params;
    params.max_diff_pixels = 32;
    params.max_num_threads = 2;
    params.max_cache_size = 512 * 1024 * 1024; // MB
    params.min_tiles_count_for_gpu_compare = 128;

    return params;
}

tile_reducer_params tile_reducer_params_make_from_args(const int argc, char** const argv);

static const char tile_reducer_param_max_diff_pixels[] = "--max-diff-pixels";
static const char tile_reducer_param_path[] = "--path";
static const char tile_reducer_param_max_cache_size[] = "--max-cache-size-mb";
static const char tile_reducer_param_max_num_threads[] = "--max-num-threads";

#endif // RUNPARAMS_H
