#include "reduce_utils.h"

#include "cache_utils.h"

#include <string.h> // for memcpy
#include <pthread.h>
#include <math.h>
#include <string.h>
#include "logging.h"

typedef struct thread_params {
    Tile** tiles_array;
    unsigned int count;
    CacheInfo* cache_info;
    reduce_results_t* reduce_results;
    tile_reducer_params params;
} thread_params;

typedef struct thread_result {
    unsigned int new_count;
} thread_result;

static inline thread_result* thread_result_new(const unsigned int new_count) {
    thread_result* tr = malloc(sizeof(thread_result));
    tr->new_count = new_count;

    return tr;
}


static inline void reduce_and_delete(Tile** const tiles_array,
                                     unsigned int* const results,
                                     const unsigned int left_tile_id,
                                     const unsigned int results_count,
                                     CacheInfo* const cache_info,
                                     const tile_reducer_params params,
                                     reduce_results_t* const reduce_results,
                                     unsigned int* const new_last_idx) {
    const Tile* t_reduced_tile = NULL;

    unsigned int* t_reduced_by = NULL;

    unsigned int e_idx = 0;

    for (unsigned int i = 0; i < results_count; ++i) {
        if (results[i] <= params.max_diff_pixels) {
            t_reduced_tile = tiles_array[i];

            t_reduced_by = find(reduce_results, t_reduced_tile->tile_id)->data;
            *t_reduced_by = left_tile_id;

            delete_image_in_cache(t_reduced_tile->tile_id, cache_info);

            tiles_array[i] = NULL;
        } else {
            if (e_idx != i) {
                tiles_array[e_idx] = tiles_array[i];
                tiles_array[i] = NULL;
            }

            e_idx++;
        }
    }

    *new_last_idx = e_idx - 1 * (e_idx != 0);
}

void reduce_tiles_array(Tile** const tiles_array,
                        const unsigned int count,
                        const unsigned int skip_index,
                        CacheInfo* const cache_info,
                        reduce_results_t* const reduce_results,
                        const tile_reducer_params params,
                        unsigned int* const new_count) {
    unsigned int* const t_results = malloc(sizeof(unsigned int) * count - 1);

    unsigned int t_tail_count = 0;

    unsigned int last_idx = skip_index >= count ? count - 1 : skip_index;

    for (unsigned int lidx = 0, ridx = 1; lidx < last_idx; ++lidx, ++ridx) {
        t_tail_count = last_idx - lidx;

        calc_diff_one_with_many(tiles_array[lidx], (const Tile* const* const)&tiles_array[ridx], t_tail_count, cache_info, params, t_results);

        delete_image_in_cache(tiles_array[lidx]->tile_id, cache_info);

        reduce_and_delete(&tiles_array[ridx], t_results, tiles_array[lidx]->tile_id, t_tail_count, cache_info, params, reduce_results, &last_idx);

        last_idx += ridx;

        tile_reducer_log_debug("left idx -> %u last idx -> %u", lidx, last_idx);
    }

    free(t_results);

    *new_count = last_idx + 1;
}

static inline unsigned int ffs_uint(const unsigned int number) {
    unsigned int v = number;

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;

    return ++v;
}

void* reduce_tiles_chunk(void* params) {
    thread_params* const tp = params;

    const unsigned int max_items_per_chunk = tp->cache_info->max_cache_size_images_nodes / TILE_SIZE_BYTES; // max_chunk_size == max tiles in cache

    const unsigned int full_chunks_by_max_items = tp->count / max_items_per_chunk; // may be 1, 2, 6, etc.
    const unsigned int full_chunks_count = ffs_uint(full_chunks_by_max_items) / 2; // may be 0, 2, 4, etc. (power of 2)


    unsigned int t_coeff = 0;

    unsigned int t_left_original_index = 0;
    unsigned int t_right_original_index = 0;

    unsigned int t_left_count = 0;
    unsigned int t_right_count = 0;

    unsigned int t_left_count_new = 0;
    unsigned int t_right_count_new = 0;

    unsigned int t_merged_count = 0;

    Tile** t_left_chunk_array = NULL;
    Tile** t_right_chunk_array = NULL;

    unsigned int full_chunk_idx_count_map[full_chunks_count];

    for (unsigned int i = 0; i < full_chunks_count; ++i) {
        full_chunk_idx_count_map[i] = max_items_per_chunk;
    }

    for (unsigned int nchunks = full_chunks_count; nchunks >= 2; nchunks /= 2) {
        t_coeff = full_chunks_count / nchunks; // for 16 => 16/16 -> 1, 16/8 -> 2, 16/4 -> 4, etc.

        for (unsigned int cidx = 0; cidx < nchunks; cidx += 2) { // for 16 => 0,2,4,6,8,10,12,14; 0,4,8,12; 0,8
            t_left_original_index = t_coeff * cidx;
            t_right_original_index = t_coeff * (cidx + 1);

            t_left_chunk_array = &tp->tiles_array[t_left_original_index * max_items_per_chunk];
            t_right_chunk_array = &tp->tiles_array[t_right_original_index * max_items_per_chunk];

            t_left_count = full_chunk_idx_count_map[t_left_original_index];
            t_right_count = full_chunk_idx_count_map[t_right_original_index];

            // reduce 2 arrays
            reduce_tiles_array(t_left_chunk_array, t_left_count, t_left_count, tp->cache_info, tp->reduce_results, tp->params, &t_left_count_new);
            reduce_tiles_array(t_right_chunk_array, t_right_count, t_right_count, tp->cache_info, tp->reduce_results, tp->params, &t_right_count_new);

            // merge their into one
            memcpy(&t_left_chunk_array[t_left_count_new], t_right_chunk_array, sizeof(Tile*) * t_right_count_new);
            memset(t_right_chunk_array, 0, sizeof(Tile*) * t_right_count_new);// TODO: remove?

            t_merged_count = t_left_count_new + t_right_count_new;

            // reduce merged array
            reduce_tiles_array(t_left_chunk_array, t_merged_count, t_left_count_new, tp->cache_info, tp->reduce_results, tp->params, &t_left_count_new);


            full_chunk_idx_count_map[t_left_original_index] = t_left_count_new;
        }
    }

    const unsigned int tail_count = tp->count - full_chunks_count * max_items_per_chunk;

    const unsigned int tail_offset = full_chunks_count * max_items_per_chunk;


    if (tail_count != 0) {
        t_left_chunk_array = &tp->tiles_array[tail_offset];

        reduce_tiles_array(t_left_chunk_array, tail_count, tail_count, tp->cache_info, tp->reduce_results, tp->params, &t_left_count_new);

        if (full_chunks_count > 0) { // merge with big one
            const unsigned int big_count = full_chunk_idx_count_map[0];

            memcpy(&tp->tiles_array[big_count], &tp->tiles_array[tail_offset], sizeof(Tile*) * t_left_count_new);
            memset(&tp->tiles_array[tail_offset], 0, sizeof(Tile*) * t_left_count_new);// TODO: remove?

            t_merged_count = big_count + t_left_count_new;

            reduce_tiles_array(tp->tiles_array, t_merged_count, big_count, tp->cache_info, tp->reduce_results, tp->params, &t_left_count_new);
        }

        return thread_result_new(t_left_count_new);
    } else { // assume we have more than zero array size
        return thread_result_new(full_chunk_idx_count_map[0]);
    }
}

reduce_results_t* reduce_tiles(Tile* const* const tiles_array,
                               const unsigned int count,
                               const tile_reducer_params params) {
    const size_t array_size = sizeof(Tile*) * count;

    Tile** const tiles_array_clone = malloc(array_size);
    memcpy(tiles_array_clone, tiles_array, array_size);

    unsigned int* t_mirror = malloc(sizeof(unsigned int));
    *t_mirror = tiles_array_clone[0]->tile_id;

    reduce_results_t* results = create_node((*t_mirror), t_mirror);

    for (unsigned int i = 1; i < count; ++i) {
        t_mirror = malloc(sizeof(unsigned int));
        (*t_mirror) = tiles_array_clone[i]->tile_id;

        results = insert(results, (*t_mirror), t_mirror);
    }



    // future
//    const unsigned char num_threads = params.max_num_threads;

//    if (num_threads > 1) {
//        pthread_t threads[num_threads];

//        for (unsigned char i = 0; i < num_threads; ++i) {
//            pthread_create(&threads[i], NULL, &reduce_tiles_chunk, &tps[i]);
//        }

//        for (unsigned char i = 0; i < num_threads - 1; ++i) {
//            pthread_join(threads[i], NULL);
//        }
//    } else {
//        const size_t diffs_cache_size = floor(params.max_cache_size * 0.05); // 5%


//        thread_params tp;
//        tp.cache_info = cache_info_new(params.max_cache_size - diffs_cache_size, diffs_cache_size, TILE_SIZE_BYTES);
//        tp.count = count;
//        tp.params = params;
//        tp.reduce_results = results;
//        tp.tiles_array = tiles_array_clone;

//        thread_result* tr = reduce_tiles_chunk(&tp);
//        free(tr);

//        cache_info_free(tp.cache_info);
//    }

    unsigned int new_count = 0;

    const size_t diffs_cache_size = floor(params.max_cache_size * 0.05); // 5%

    CacheInfo* const cache_info = cache_info_new(params.max_cache_size - diffs_cache_size, diffs_cache_size, TILE_SIZE_BYTES);
    reduce_tiles_array(tiles_array_clone, count, count, cache_info, results, params, &new_count);

    cache_info_free(cache_info);

    free(tiles_array_clone);

    return results;
}
