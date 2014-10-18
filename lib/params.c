#include "params.h"

#include <string.h>
#include <stdlib.h>

#define IS_STRINGS_EQUAL(str0, st1) (strcmp(str0, st1) == 0)
#define IS_STRINGS_NOT_EQUAL(str0, st1) (strcmp(str0, st1) != 0)

#define PARAM_NOT_FOUND "false"
#define PARAM_FOUND "true"

#define IS_PARAM_FOUND(param) (IS_STRINGS_EQUAL(param, PARAM_NOT_FOUND) == 0)
#define IS_PARAM_NOT_FOUND(param) IS_STRINGS_EQUAL(param, PARAM_NOT_FOUND)


char* get_arg(const int argc, char** argv, const char* const key) {
    int i = 0;

    for(; i < argc; ++i) {
        if (strstr(argv[i], key) != NULL) {
            if (strchr(argv[i], '=') != NULL) {
                return &(argv[i][strlen(key) + 1]);
            } else {
                return PARAM_FOUND;
            }
        }
    }

    return PARAM_NOT_FOUND;
}

tile_reducer_params tile_reducer_params_make_from_args(const int argc, char** const argv) {
    tile_reducer_params arp = tile_reducer_params_make_default();

    strcpy(arp.path, get_arg(argc, argv, tile_reducer_param_path));

    const char* const max_diff_pixels_param = get_arg(argc, argv, tile_reducer_param_max_diff_pixels);

    if (IS_PARAM_NOT_FOUND(max_diff_pixels_param)) {
        arp.max_diff_pixels = atoi(max_diff_pixels_param);
    }

    const char* const max_cache_size_param = get_arg(argc, argv, tile_reducer_param_max_cache_size);

    if (IS_PARAM_FOUND(max_cache_size_param)) {
        arp.max_cache_size = ((size_t) atoi(max_cache_size_param)) * 1024 * 1024;
    }

    const char* const max_num_threads_param = get_arg(argc, argv, tile_reducer_param_max_num_threads);

    if (IS_PARAM_FOUND(max_num_threads_param)) {
        arp.max_num_threads = atoi(max_num_threads_param);
    }

    return arp;
}
