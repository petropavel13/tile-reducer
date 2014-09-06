#include "tile_utils.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <pthread.h>
#include <dirent.h>
#include "lodepng.h"
#include "gpu_utils.h"

#define MIN_TILES_PER_THREAD 256
#define MIN_TILES_FOR_MULTITHREADING (MIN_TILES_PER_THREAD * 2)

typedef struct CompareBackend {
    unsigned int (*one_with_one_func)(const unsigned char* const,
                                        const unsigned char* const);

    TaskStatus (*one_with_many_func)(const unsigned char* const,
                               const unsigned char* const,
                               const unsigned int,
                               unsigned int* const);

    void* (*memory_allocator)(size_t);
    void (*memory_deallocator)(void*);
} CompareBackend;

typedef enum CompareBackendType { CPU, CUDA_GPU } CompareBackendType;

typedef struct LoadTilesParams {
    Tile** tiles;
    unsigned int count;
    unsigned char** raw_output;
    unsigned char** raw_cache_output;
} LoadTilesParams;



unsigned int compare_images_one_with_one_cpu(const unsigned char * const raw_left_image,
                                               const unsigned char * const raw_right_image) {
    unsigned int res = 0;

    for (unsigned int i = 0; i < TILE_SIZE; i += 4) {
        res += (raw_left_image[i+0] != raw_right_image[i+0] || // r
                raw_left_image[i+1] != raw_right_image[i+1] || // g
                raw_left_image[i+2] != raw_right_image[i+2] || // b
                raw_left_image[i+3] != raw_right_image[i+3]);  // a
    }

    return res;
}

TaskStatus compare_images_one_with_many_cpu(const unsigned char* const left_raw_image,
                                            const unsigned char* const right_raw_images,
                                            const unsigned int right_images_count,
                                            unsigned int* const diff_results) {
    for (unsigned int i = 0; i < right_images_count; ++i) {
        diff_results[i] = compare_images_one_with_one_cpu(left_raw_image, &right_raw_images[i * TILE_SIZE_BYTES]);
    }

    return TASK_DONE;
}

void* cpu_backend_memory_allocator(size_t bytes) {
    return malloc(bytes);
}

void cpu_backend_memory_deallocator(void* ptr) {
    free(ptr);
}

CompareBackend make_backend(CompareBackendType type, const unsigned int count) {
    CompareBackend cb = { NULL, NULL, NULL, NULL };

    if (type == CPU) {
        cb.one_with_one_func = &compare_images_one_with_one_cpu;
        cb.one_with_many_func = &compare_images_one_with_many_cpu;

        cb.memory_allocator = &cpu_backend_memory_allocator;
        cb.memory_deallocator = &cpu_backend_memory_deallocator;
    } else if(type == CUDA_GPU) {
        cb.memory_allocator = &gpu_backend_memory_allocator;
        cb.memory_deallocator = &gpu_backend_memory_deallocator;

        if (count < get_max_tiles_count_per_stream()) {
            cb.one_with_many_func = &compare_one_image_with_others;
        } else {
            cb.one_with_many_func = &compare_one_image_with_others_streams;
        }
    }

    return cb;
}

TileFile* read_tile(const char* file_path) {
    TileFile* const tile = malloc(sizeof(TileFile));

    lodepng_load_file(&tile->data, &tile->size_bytes, file_path);

    return tile;
}


unsigned int get_tile_pixels(const TileFile* const tile, unsigned char** const pixels) {
    unsigned int width, height;

    return lodepng_decode32(pixels, &width, &height, tile->data, tile->size_bytes);
}

void tile_file_destructor(TileFile* tile_file) {
    free(tile_file->data);
    free(tile_file);
}

void load_tiles_pixels_threads(const Tile* const * const tiles,
                               const unsigned int count,
                               CacheInfo* const cache_info,
                               const AppRunParams arp,
                               unsigned char* const raw_tiles) {
    const Tile* tile_for_decode[count];
    unsigned int index_mapping[count];
    unsigned int count_for_decode = 0;

    const Tile* t_tile = NULL;

    unsigned char* t_raw_tile = NULL;

    for (unsigned int i = 0; i < count; ++i) {
        t_tile = tiles[i];

        if (get_tile_data(t_tile->tile_id, cache_info, &t_raw_tile) == CACHE_MISS) {
            index_mapping[count_for_decode] = i;
            tile_for_decode[count_for_decode++] = t_tile;
        } else {
            memcpy(&(raw_tiles[i * TILE_SIZE]), t_raw_tile, TILE_SIZE_BYTES);
        }
    }

    if (count_for_decode < MIN_TILES_FOR_MULTITHREADING) {
        for (unsigned int i = 0; i < count_for_decode; ++i) {
            t_tile = tile_for_decode[i];

            if(get_tile_pixels(t_tile->tile_file, &t_raw_tile) != 0) {
                printf("\n\nproblem while loading tile with id: %d!\n\n", t_tile->tile_id);
                fflush(stdout);
            } else {
                memcpy(&(raw_tiles[index_mapping[i] * TILE_SIZE]), t_raw_tile, TILE_SIZE_BYTES);
            }

            push_image_to_cache(t_tile->tile_id, t_raw_tile, cache_info);
        }
    } else {
        const unsigned int t_th_count = floor((double)count / (double)MIN_TILES_PER_THREAD);
        const unsigned int num_threads = t_th_count > arp.max_num_threads ? arp.max_num_threads : t_th_count;

        const unsigned int tiles_per_thread = floor((double)count_for_decode / (double)num_threads);

        pthread_t threads[num_threads];
        LoadTilesParams ltps[num_threads];

        unsigned int t_offset = 0;

        const Tile* tiles_for_thread_decode[count_for_decode];
        unsigned char* raw_tiles_links[count_for_decode];

        for (unsigned int i = 0; i < count_for_decode; ++i) {
            tiles_for_thread_decode[i] = tiles[index_mapping[i]];
            raw_tiles_links[i] = &raw_tiles[index_mapping[i] * TILE_SIZE];
        }

        unsigned char* cache_out[count_for_decode];

        for (unsigned char i = 0; i < num_threads - 1; ++i) {
            ltps[i] = (LoadTilesParams) {
                    .tiles = (Tile**)tiles_for_thread_decode + t_offset,
                    .count = tiles_per_thread,
                    .raw_output = raw_tiles_links + t_offset,
                    .raw_cache_output = &cache_out[t_offset],
                };
            t_offset += tiles_per_thread;
        }

        // last thread has more work
        ltps[num_threads - 1] = (LoadTilesParams) {
                .tiles = (Tile**)tiles_for_thread_decode + t_offset,
                .count = count_for_decode - t_offset,
                .raw_output = raw_tiles_links + t_offset,
                .raw_cache_output = &cache_out[t_offset],
            };

        for (unsigned char i = 0; i < num_threads; ++i) {
            pthread_create(&threads[i], NULL, &load_tiles_pixels_part, &ltps[i]);
        }

        for (unsigned char i = 0; i < num_threads - 1; ++i) {
            pthread_join(threads[i], NULL);

            t_offset = i * tiles_per_thread;

            for (unsigned int j = 0; j < tiles_per_thread; ++j) {
                push_image_to_cache(tiles[index_mapping[t_offset + j]]->tile_id, cache_out[t_offset + j], cache_info);
            }
        }

        pthread_join(threads[num_threads - 1], NULL);

        t_offset = (num_threads - 1) * tiles_per_thread;

        for (unsigned int j = 0; j < count_for_decode - t_offset; ++j) {
            push_image_to_cache(tiles[index_mapping[t_offset + j]]->tile_id, cache_out[t_offset + j], cache_info);
        }
    }
}

void* load_tiles_pixels_part(void* params) {
    const LoadTilesParams* const p = params;

    unsigned char* t_pixels = NULL;

    Tile* t_tile = NULL;

    for (unsigned int i = 0; i < p->count; ++i) {
        t_tile = p->tiles[i];

        if(get_tile_pixels(t_tile->tile_file, &t_pixels) != 0) {
            printf("\n\nproblem while loading tile with id: %d!\n\n", t_tile->tile_id);
            fflush(stdout);
        } else {
            memcpy(p->raw_output[i], t_pixels, TILE_SIZE_BYTES);
            p->raw_cache_output[i] = t_pixels;
        }
    }

    return NULL;
}

void load_pixels(const Tile* const tile,
                 CacheInfo* const cache_info,
                 unsigned char ** const pixels) {
    unsigned char* t_pixels = NULL;

    if(get_tile_data(tile->tile_id, cache_info, &t_pixels) == CACHE_MISS) {
        if(get_tile_pixels(tile->tile_file, &t_pixels) == 0) {
            push_image_to_cache(tile->tile_id, t_pixels, cache_info);
        } else {
            printf("\n\nproblem while loading tile with id: %d!\n\n", tile->tile_id);
            fflush(stdout);
        }
    }

    *pixels = malloc(TILE_SIZE_BYTES);

    memcpy(*pixels, t_pixels, TILE_SIZE_BYTES);
}

unsigned int calc_diff(const Tile* const left_node,
                             const Tile* const right_node,
                             CacheInfo* const cache_info) {
    unsigned int diff_result;
    const unsigned long key = make_key(left_node->tile_id, right_node->tile_id);

    if(get_diff_from_cache(key, cache_info, &diff_result) == CACHE_HIT) {
        return diff_result;
    }

    unsigned char* left_tile_pixels = NULL;
    load_pixels(left_node, cache_info, &left_tile_pixels);

    unsigned char* right_tile_pixels = NULL;
    load_pixels(right_node, cache_info, &right_tile_pixels);

    const CompareBackend cb = make_backend(CPU, 1);

    diff_result = cb.one_with_one_func(left_tile_pixels, right_tile_pixels);
    push_edge_to_cache(key, diff_result, cache_info);

    free(left_tile_pixels);
    free(right_tile_pixels);

    return diff_result;
}

void calc_diff_one_with_many(const Tile* const left_tile,
                             const Tile * const *const right_tiles,
                             const unsigned int right_tiles_count,
                             CacheInfo* const cache_info,
                             const AppRunParams arp,
                             unsigned int* const results) {
    const Tile* right_tiles_for_load[right_tiles_count];
    unsigned long keys_for_load[right_tiles_count];
    unsigned int index_mapping[right_tiles_count];

    unsigned int count_for_load = 0;

    unsigned long t_key = 0;

    for (unsigned int i = 0; i < right_tiles_count; ++i) {
        t_key = make_key(left_tile->tile_id, right_tiles[i]->tile_id);

        if (get_diff_from_cache(t_key, cache_info, &results[i]) == CACHE_MISS) {
            right_tiles_for_load[count_for_load] = right_tiles[i];
            keys_for_load[count_for_load] = t_key;
            index_mapping[count_for_load++] = i;
        }
    }

    const unsigned int max_tiles_per_mem_loop = floor((double)cache_info->max_cache_size_images_nodes / (double)2 / (double)TILE_SIZE_BYTES);

    const unsigned int tiles_per_mem_loop = count_for_load < max_tiles_per_mem_loop ? count_for_load : max_tiles_per_mem_loop;

    const unsigned int loops_count = floor((double)count_for_load / (double)tiles_per_mem_loop);

    const unsigned int tail_count = count_for_load - loops_count * tiles_per_mem_loop;

    unsigned int t_offset = 0;

    unsigned char* left_tile_pixels = NULL;
    load_pixels(left_tile, cache_info, &left_tile_pixels);

    CompareBackend t_cb = make_backend(count_for_load >= arp.min_tiles_count_for_gpu_compare ? CUDA_GPU : CPU, tiles_per_mem_loop);

    unsigned char* t_raw_right_tiles = t_cb.memory_allocator(TILE_SIZE_BYTES * tiles_per_mem_loop);

    unsigned int* const t_results = malloc(sizeof(unsigned int) * tiles_per_mem_loop);

    for (unsigned int i = 0; i < loops_count; ++i) {
        load_tiles_pixels_threads((const Tile* const * const)right_tiles_for_load + t_offset, tiles_per_mem_loop, cache_info, arp, t_raw_right_tiles);

        if (t_cb.one_with_many_func(left_tile_pixels, t_raw_right_tiles, tiles_per_mem_loop, t_results) == TASK_FAILED) {
            // when error occur cudaDeviceReset destroy all associated resources
            // so we need to call this function again
            t_cb.memory_deallocator(t_raw_right_tiles);
            calc_diff_one_with_many(left_tile, right_tiles,right_tiles_count, cache_info, arp, results);
            return;
        }

        for (unsigned int j = 0; j < tiles_per_mem_loop; ++j) {
            push_edge_to_cache(keys_for_load[t_offset + j], t_results[j], cache_info);
            results[index_mapping[t_offset + j]] = t_results[j];
        }

        t_offset += tiles_per_mem_loop;
    }

    t_cb.memory_deallocator(t_raw_right_tiles);


    if (tail_count > 0) {
        if (tail_count == 1) {
            t_cb = make_backend(CPU, tail_count);

            unsigned char* right_tile_pixels = NULL;
            load_pixels(right_tiles[t_offset], cache_info, &right_tile_pixels);

            t_results[t_offset] = t_cb.one_with_one_func(left_tile_pixels, right_tile_pixels);

            push_edge_to_cache(keys_for_load[t_offset], t_results[t_offset], cache_info);
            results[index_mapping[t_offset]] = t_results[t_offset];

            free(right_tile_pixels);
        } else {
            t_cb = make_backend(tail_count >= arp.min_tiles_count_for_gpu_compare ? CUDA_GPU : CPU, tail_count);

            t_raw_right_tiles = t_cb.memory_allocator(TILE_SIZE_BYTES * tail_count);

            load_tiles_pixels_threads((const Tile* const * const)right_tiles_for_load + t_offset, tail_count, cache_info, arp, t_raw_right_tiles);

            if (t_cb.one_with_many_func(left_tile_pixels, t_raw_right_tiles, tail_count, t_results) == TASK_FAILED) {
                // when error occur cudaDeviceReset destroy all associated resources
                // so we need to call this function again
                t_cb.memory_deallocator(t_raw_right_tiles);
                calc_diff_one_with_many(left_tile, right_tiles,right_tiles_count, cache_info, arp, results);
                return;
            }

            for (unsigned int i = 0; i < tail_count; ++i) {
                push_edge_to_cache(keys_for_load[t_offset + i], t_results[i], cache_info);
                results[index_mapping[t_offset + i]] = t_results[i];
            }

            t_cb.memory_deallocator(t_raw_right_tiles);
        }
    }

    free(left_tile_pixels);
}

void tile_destructor(void* data) {
    Tile* const t = data;
    tile_file_destructor(t->tile_file);
    free(t);
}


