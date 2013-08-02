#ifndef TILE_UTILS_H
#define TILE_UTILS_H

#include <dirent.h>
#include "lodepng.h"
#include <stdlib.h>
#include <stdio.h>
#include "cache_utils.h"
#include "gpu_utils.h"

#define TILE_WIDTH 256
#define TILE_HEIGHT 256

#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT * 4)
#define TILE_SIZE_BYTES (TILE_SIZE * sizeof(unsigned char))

extern unsigned int compare_images(unsigned char* raw_left_image,
                                   unsigned char* raw_right_image,
                                   unsigned short int* diff_result);

typedef struct TileFile
{
    unsigned char* data;
    size_t size_bytes;
} TileFile;


typedef struct Tile
{
    unsigned int tile_id;
    TileFile* tile_file;
} Tile;




TileFile* read_tile(const char* file_path);

unsigned int get_tile_pixels(const TileFile* tile, unsigned char** pixels);

unsigned int get_total_files_count(const char* path);

void read_tiles_paths(const char* path,
                      char** paths,
                      unsigned int* current,
                      void (*callback)(unsigned int, const char*, const char*));

void delete_tile_file(TileFile* tile_file);

unsigned int calc_diff(const Tile* const left_node, const Tile* const right_node, CacheInfo* const cache_info);

#endif // TILE_UTILS_H
