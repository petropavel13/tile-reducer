#ifndef TILE_UTILS_H
#define TILE_UTILS_H

#include "cache_utils.h"
#include "apprunparams.h"

#define TILE_WIDTH 256
#define TILE_HEIGHT 256
#define TILE_DEEP 4

#define TILE_PIXELS_COUNT TILE_WIDTH * TILE_HEIGHT
#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT * TILE_DEEP)
#define TILE_SIZE_BYTES (TILE_SIZE * sizeof(unsigned char))

typedef struct TileFile {
    unsigned char* data;
    size_t size_bytes;
} TileFile;


typedef struct Tile {
    unsigned int tile_id;
    TileFile* tile_file;
} Tile;


TileFile* read_tile(const char* absolute_file_path);

unsigned int get_tile_pixels(const TileFile* const tile, unsigned char** const pixels);

void tile_file_destructor(TileFile* tile_file);
void tile_destructor(void* data);

void load_tiles_pixels_threads(const Tile* const * const tiles,
                               const unsigned int count,
                               CacheInfo* const cache_info,
                               const AppRunParams arp,
                               unsigned char* const raw_tiles);

void load_pixels(const Tile* const tile,
                 CacheInfo* const cache_info,
                 unsigned char** const pixels);

unsigned int calc_diff(const Tile* const left_node,
                             const Tile* const right_node,
                             CacheInfo *const cache_info);

void calc_diff_one_with_many(const Tile* const left_tile,
                             const Tile* const * const right_tiles,
                             const unsigned int right_tiles_count,
                             CacheInfo* const cache_info,
                             const AppRunParams arp,
                             unsigned int* const results);


#endif // TILE_UTILS_H
