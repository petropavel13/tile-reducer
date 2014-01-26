#include "tile_utils.h"

TileFile* read_tile(const char* file_path) {
    unsigned char* file;
    size_t file_size;

    lodepng_load_file(&file, &file_size, file_path);

    TileFile* tile = malloc(sizeof(TileFile));
    tile->data = file;
    tile->size_bytes = file_size;

    return tile;
}


unsigned int get_tile_pixels(const TileFile* const tile, unsigned char** const pixels) {
    unsigned int width, height;

    return lodepng_decode32(pixels, &width, &height, tile->data, tile->size_bytes);
}


unsigned int get_total_files_count(const char* const path) {
    DIR* dir = NULL;
    struct dirent *entry;

    unsigned int count = 0;

    if((dir = opendir(path)) != NULL) {
        while ((entry = readdir(dir)) != 0) {
            if(entry->d_type & DT_DIR) { // directory
                if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                    continue;

                char inner_path[strlen(path) + strlen(entry->d_name + 1)];
                sprintf(inner_path, "%s%s/", path, entry->d_name);

                count += get_total_files_count(inner_path);
            } else {
                count++;
            }
        }

        closedir(dir);
    }

    return count;
}

void read_tiles_paths(const char* path,
                      char** const paths,
                      const unsigned int *const total,
                      unsigned int *const current,
                      unsigned char *const last_percent,
                      void (*callback)(unsigned char)) {
    DIR* dir = NULL;
    struct dirent *entry;

    if((dir = opendir(path)) != NULL) {
        while ((entry = readdir(dir)) != 0) {
            if(entry->d_type & DT_DIR) {
                if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                    continue;

                char inner_path[strlen(path) + strlen(entry->d_name) + 1 + 1];
                sprintf(inner_path, "%s%s/", path, entry->d_name);

                read_tiles_paths(inner_path, paths, total, current, last_percent, callback);
            } else {
                char* const inner_path = malloc(sizeof(char) * (strlen(path) + strlen(entry->d_name) + 1));
                sprintf(inner_path, "%s%s", path, entry->d_name);

                paths[(*current)++] = inner_path;

                if(callback != NULL) {
                    const unsigned char current_percent = (*current / (*total / 100));

                    if(*last_percent != current_percent) {
                        callback(current_percent);
                        *last_percent = current_percent;
                    }
                }
            }
        }
        
        closedir(dir);
    }
}

void tile_file_destructor(TileFile* tile_file) {
    free(tile_file->data);
    free(tile_file);
}

unsigned short compare_images_one_with_one_cpu(const unsigned char * const raw_left_image,
                                               const unsigned char * const raw_right_image) {
    unsigned int res = 0;

    for (unsigned int i = 0; i < TILE_SIZE; i += 4) {
        res += (raw_left_image[i+0] != raw_right_image[i+0] ||
                raw_left_image[i+1] != raw_right_image[i+1] ||
                raw_left_image[i+2] != raw_right_image[i+2] ||
                raw_left_image[i+3] != raw_right_image[i+3]);
    }

    return (unsigned short) (res > USHORT_MAX ? USHORT_MAX : res);
}

TaskStatus compare_images_one_with_many_cpu(const unsigned char * const left_raw_image,
                                      const unsigned char * const right_raw_images,
                                      const unsigned int right_images_count,
                                      unsigned short * const diff_results) {
    for (unsigned int i = 0; i < right_images_count; ++i) {
        diff_results[i] = compare_images_one_with_one_cpu(left_raw_image, &right_raw_images[i * TILE_SIZE_BYTES]);
    }

    return TASK_DONE;
}

void load_pixels(const Tile* const tile,
                 CacheInfo* const cache_info,
                 unsigned char ** const pixels) {
    const CacheSearchResult cache_res = get_tile_data(tile->tile_id, cache_info, pixels);

    if(cache_res == CACHE_MISS) {
        const unsigned int read_res = get_tile_pixels(tile->tile_file, pixels);

        if(read_res == 0) {
            push_image_to_cache(tile->tile_id, *pixels, cache_info);
        } else {
            printf("\n\nproblem while loading tile with id: %d!\n\n", tile->tile_id);
            fflush(stdout);
        }
    }
}

unsigned int calc_diff(const Tile* const left_node,
                       const Tile* const right_node,
                       CacheInfo* const cache_info) {
    unsigned short diff_result;
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

    return diff_result;
}

void calc_diff_one_with_many(const Tile* const left_tile,
                             const Tile * const *const right_tiles,
                             const unsigned int right_tiles_count,
                             CacheInfo* const cache_info,
                             unsigned short int * const results) {
    unsigned char* left_tile_pixels = NULL;
    load_pixels(left_tile, cache_info, &left_tile_pixels);

    unsigned int rest_count = right_tiles_count;
    unsigned int current = 0;

    unsigned char* right_tiles_pixels = (unsigned char*)malloc(TILE_SIZE_BYTES * TILE_SIZE_BUFFER);

    unsigned char* temp_right_tile_pixels = NULL;

    TaskStatus status = 0;

    const CompareBackend cb = make_backend(right_tiles_count > 32 ? CUDA_GPU : CPU, right_tiles_count);

    while (rest_count > TILE_SIZE_BUFFER) {
        for (unsigned int i = 0; i < TILE_SIZE_BUFFER; ++i) {
            load_pixels(right_tiles[current + i], cache_info, &temp_right_tile_pixels);
            memcpy(&right_tiles_pixels[i * TILE_SIZE_BYTES], temp_right_tile_pixels, TILE_SIZE_BYTES);
        }

        status = cb.one_with_many_func(left_tile_pixels, right_tiles_pixels, TILE_SIZE_BUFFER, &(results[current]));
//        status = compare_one_image_with_others(left_tile_pixels, right_tiles_pixels, TILE_SIZE_BUFFER, &(results[current]));

        if(status == TASK_FAILED) {
            printf("\n\nTASK FAILED!\n\n");
            fflush(stdout);

            return;
        } else {
            for (unsigned int j = 0; j < TILE_SIZE_BUFFER; ++j) {
                push_edge_to_cache(make_key(left_tile->tile_id, right_tiles[j]->tile_id), results[current + j], cache_info);
            }
        }

        rest_count -= TILE_SIZE_BUFFER;
        current += TILE_SIZE_BUFFER;
    }

    for (unsigned int i = 0; i < rest_count; ++i) {
        load_pixels(right_tiles[current + i], cache_info, &temp_right_tile_pixels);
        memcpy(&right_tiles_pixels[i * TILE_SIZE_BYTES], temp_right_tile_pixels, TILE_SIZE_BYTES);
    }

    status = cb.one_with_many_func(left_tile_pixels, right_tiles_pixels, rest_count, &(results[current]));
//    status = compare_one_image_with_others(left_tile_pixels, right_tiles_pixels, rest_count, &(results[current]));

    if(status == TASK_FAILED) {
        printf("\n\nTASK FAILED!\n\n");
        fflush(stdout);

        return;
    } else {
        for (unsigned int j = 0; j < rest_count; ++j) {
            push_edge_to_cache(make_key(left_tile->tile_id, right_tiles[j]->tile_id), results[current + j], cache_info);
        }
    }

    free(right_tiles_pixels);
}

void tile_destructor(void* data) {
    Tile* const t = data;
    tile_file_destructor(t->tile_file);
    free(t);
}


CompareBackend make_backend(CompareBackendType type, const unsigned int count) {
    CompareBackend cb = { NULL, NULL };

    if (type == CPU) {
        cb.one_with_one_func = &compare_images_one_with_one_cpu;
        cb.one_with_many_func = &compare_images_one_with_many_cpu;
    } else if(type == CUDA_GPU) {
//        cb.one_with_one_func = &compare_one_image_with_others;

        if (count < get_max_tiles_count_per_stream()) {
            cb.one_with_many_func = &compare_one_image_with_others;
        } else {
            cb.one_with_many_func = &compare_one_image_with_others_streams;
        }
    }

    return cb;
}



