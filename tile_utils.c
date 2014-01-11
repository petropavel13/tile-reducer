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


unsigned int get_tile_pixels(const TileFile* tile, unsigned char** pixels) {
    unsigned int width, height;

    return lodepng_decode32(pixels, &width, &height, tile->data, tile->size_bytes);
}


unsigned int get_total_files_count(const char* path) {
    DIR* dir = NULL;
    struct dirent *entry;

    unsigned int count = 0;

    if((dir = opendir(path)) != NULL) {
        while ((entry = readdir(dir)) != 0) {
            if(entry->d_type & DT_DIR) { // directory
                if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                    continue;

                char inner_path[255];
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
                char* inner_path = malloc(sizeof(char) * (strlen(path) + strlen(entry->d_name) + 1));
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

void delete_tile_file(TileFile* tile_file) {
    free(tile_file->data);
    free(tile_file);
}

unsigned short compare_images_cpu(unsigned char * const raw_left_image, unsigned char * const raw_right_image) {
    unsigned int res = 0;

    for (unsigned int i = 0; i < TILE_SIZE; i += 4) {
        res += (raw_left_image[i+0] != raw_right_image[i+0] ||
                raw_left_image[i+1] != raw_right_image[i+1] ||
                raw_left_image[i+2] != raw_right_image[i+2] ||
                raw_left_image[i+3] != raw_right_image[i+3]);
    }

    return (unsigned short) (res > USHORT_MAX ? USHORT_MAX : res);
}

void load_pixels(const Tile* const tile,
                 CacheInfo* const cache_info,
                 unsigned char **pixels) {
    const unsigned char cache_res = get_tile_data(tile->tile_id, cache_info, pixels);

    if(cache_res == CACHE_MISS) {
        const unsigned int read_res = get_tile_pixels(tile->tile_file, pixels);

        if(read_res == 0) {
            push_image_to_cache(tile->tile_id, *pixels, cache_info);
        } else {
            printf("\n\nproblem while loading tile with id: %d\n\n", tile->tile_id);
            fflush(stdout);
        }
    }
}

unsigned int calc_diff(const Tile* const left_node,
                       const Tile* const right_node,
                       CacheInfo* const cache_info) {
    unsigned short diff_result;
    const unsigned long key = make_key(left_node->tile_id, right_node->tile_id);

    const unsigned char cache_res = get_diff_from_cache(key, cache_info, &diff_result);

    if(cache_res  == CACHE_HIT) {
        return diff_result;
    }

    unsigned char* left_tile_pixels = NULL;
    load_pixels(left_node, cache_info, &left_tile_pixels);

    unsigned char* right_tile_pixels = NULL;
    load_pixels(right_node, cache_info, &right_tile_pixels);

    diff_result = compare_images_cpu(left_tile_pixels, right_tile_pixels);
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

    unsigned int compare_res = 0;

    while (rest_count > TILE_SIZE_BUFFER) {
        for (unsigned int i = 0; i < TILE_SIZE_BUFFER; ++i) {
            load_pixels(right_tiles[current + i], cache_info, &temp_right_tile_pixels);
            memcpy(&right_tiles_pixels[i * TILE_SIZE_BYTES], temp_right_tile_pixels, TILE_SIZE_BYTES);
        }

        compare_res = compare_one_image_with_others(left_tile_pixels, right_tiles_pixels, TILE_SIZE_BUFFER, &(results[current]));

        if(compare_res == TASK_FAILED) {
            printf("\n\nGPU TASK FAILED\n\n");
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

    compare_res = compare_one_image_with_others(left_tile_pixels, right_tiles_pixels, rest_count, &(results[current]));

    if(compare_res == TASK_FAILED) {
        printf("\n\nGPU TASK FAILED\n\n");
        fflush(stdout);

        return;
    } else {
        for (unsigned int j = 0; j < rest_count; ++j) {
            push_edge_to_cache(make_key(left_tile->tile_id, right_tiles[j]->tile_id), results[current + j], cache_info);
        }
    }

    free(right_tiles_pixels);
}
