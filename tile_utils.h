#ifndef TILE_UTILS_H
#define TILE_UTILS_H

#include <dirent.h>
#include "lodepng.h"
#include <stdlib.h>
#include <stdio.h>
#include "cache_utils.h"
#include "gpu_utils.h"
#include "apprunparams.h"

#define TILE_WIDTH 256
#define TILE_HEIGHT 256
#define TILE_DEEP 4

#define TILE_SIZE (TILE_WIDTH * TILE_HEIGHT * TILE_DEEP)
#define TILE_SIZE_BYTES (TILE_SIZE * sizeof(unsigned char))

#define USHORT_MAX 65535

typedef struct TileFile {
    unsigned char* data;
    size_t size_bytes;
} TileFile;


typedef struct Tile {
    unsigned int tile_id;
    TileFile* tile_file;
} Tile;


TileFile* read_tile(const char* file_path);

unsigned int get_tile_pixels(const TileFile * const tile, unsigned char** const pixels);

unsigned int get_total_files_count(const char * const path);

void read_tiles_paths(const char* path,
                      char **const paths,
                      const unsigned int* const total,
                      unsigned int *const current,
                      unsigned char *const last_percent,
                      void (*callback)(unsigned char));

void tile_file_destructor(TileFile* tile_file);
void tile_destructor(void* data);

void load_tiles_pixels_threads(const Tile * const * const tiles,
                               const unsigned int count,
                               CacheInfo * const cache_info,
                               const AppRunParams arp,
                               unsigned char * const raw_tiles);

void* load_tiles_pixels_part(void* params);

unsigned short compare_images_one_with_one_cpu(const unsigned char * const raw_left_image,
                                               const unsigned char * const raw_right_image);

TaskStatus compare_images_one_with_many_cpu(const unsigned char* const left_raw_image,
                                            const unsigned char* const right_raw_images,
                                            const unsigned int right_images_count,
                                            unsigned short* const diff_results);

void* cpu_backend_memory_allocator(size_t bytes);
void cpu_backend_memory_deallocator(void* ptr);

void load_pixels(const Tile* const tile,
                 CacheInfo* const cache_info,
                 unsigned char ** const pixels);

unsigned short int calc_diff(const Tile* const left_node,
                             const Tile* const right_node,
                             CacheInfo *const cache_info);

void calc_diff_one_with_many(const Tile* const left_tile,
                             const Tile *const*const right_tiles,
                             const unsigned int right_tiles_count,
                             CacheInfo* const cache_info,
                             const AppRunParams arp,
                             unsigned short *const results);


#endif // TILE_UTILS_H
