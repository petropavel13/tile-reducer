#ifndef TILE_UTILS_H
#define TILE_UTILS_H

#include "cache_utils.h"
#include "params.h"

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
    char* file_path; // may be NULL

    TileFile* tile_file;
} Tile;


TileFile* read_tile(const char* absolute_file_path);

void get_tile_pixels(const TileFile* const tile, unsigned char** const pixels);

static inline TileFile tile_file_make(unsigned char* const data, const size_t size_bytes) {
    TileFile tf;
    tf.data = data;
    tf.size_bytes = size_bytes;

    return tf;
}

static inline TileFile* tile_file_new(unsigned char* const data, const size_t size_bytes) {
    TileFile* const tf = (TileFile*)malloc(sizeof(TileFile));
    tf->data = data;
    tf->size_bytes = size_bytes;

    return tf;
}

static inline void tile_file_free(TileFile* const tile_file) {
    free(tile_file->data);
    free(tile_file);
}

static inline Tile* tile_new(TileFile* const tile_file,
                             const unsigned int tile_id,
                             char* const file_path) {
    Tile* const t = (Tile*)malloc(sizeof(Tile));
    t->tile_file = tile_file;
    t->tile_id = tile_id;
    t->file_path = file_path;

    return t;
}

void tile_free(void* data);

void load_tiles_pixels_threads(const Tile* const* const tiles,
                               const unsigned int count,
                               CacheInfo* const cache_info,
                               const tile_reducer_params arp,
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
                             const tile_reducer_params arp,
                             unsigned int* const results);


#endif // TILE_UTILS_H
