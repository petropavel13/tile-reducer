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
                      char** paths,
                      unsigned int* current,
                      void (*callback)(unsigned int)) {
    DIR* dir = NULL;
    struct dirent *entry;

    if((dir = opendir(path)) != NULL) {
        while ((entry = readdir(dir)) != 0) {
            if(entry->d_type & DT_DIR) {
                if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                    continue;

                char inner_path[strlen(path) + strlen(entry->d_name) + 1 + 1];
                sprintf(inner_path, "%s%s/", path, entry->d_name);

                read_tiles_paths(inner_path, paths, current, callback);
            } else {
                char* inner_path = malloc(sizeof(char) * (strlen(path) + strlen(entry->d_name) + 1));
                sprintf(inner_path, "%s%s", path, entry->d_name);

                paths[(*current)++] = inner_path;

                if(callback != NULL) {
                    callback(*current);
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

unsigned int calc_diff(const Tile* const left_node, const Tile* const right_node, CacheInfo* const cache_info) {
    unsigned char* left_tile_pixels = NULL;

    unsigned int cache_res;

    cache_res = get_tile_data(left_node->tile_id, cache_info, &left_tile_pixels);

    if(cache_res == CACHE_MISS) {
        unsigned int read_res = get_tile_pixels(left_node->tile_file, &left_tile_pixels);

        if(read_res != 0) {
            printf("\n\nproblem while loading tile with id: %d\n\n", right_node->tile_id);
            fflush(stdout);
        } else {
            push_image_to_cache(left_node->tile_id, left_tile_pixels, cache_info);
        }
    }

    unsigned char* right_tile_pixels = NULL;

    cache_res = get_tile_data(right_node->tile_id, cache_info, &right_tile_pixels);

    if(cache_res == CACHE_MISS) {
        unsigned int read_res = get_tile_pixels(right_node->tile_file, &right_tile_pixels);

        if(read_res != 0) {
            printf("\n\nproblem while loading tile with id: %d\n\n", right_node->tile_id);
            fflush(stdout);
        } else {
            push_image_to_cache(right_node->tile_id, right_tile_pixels, cache_info);
        }
    }

    unsigned short diff_result;

    unsigned int compare_res = compare_images(left_tile_pixels, right_tile_pixels, &diff_result);

    if(compare_res == TASK_FAILED) {
        printf("\n\nGPU TASK FAILED\n\n");
        fflush(stdout);

        return -1;
    }

    return diff_result;
}

