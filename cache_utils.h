#ifndef CACHE_UTILS_H
#define CACHE_UTILS_H

#include <stdlib.h>
#include "generic_avl_tree.h"

typedef struct CacheInfo
{
    size_t max_cache_size_images_nodes;
    size_t max_cache_size_edge_nodes;

    size_t tile_size_bytes;

    GenericNode* images_root_node;
    TreeInfo* images_tree_info;

    GenericNode* edges_root_node;
    TreeInfo* edges_tree_info;

    unsigned int images_nodes_in_cache;
    unsigned int image_hit_count;
    unsigned int image_miss_count;

    unsigned int edges_nodes_in_cache;
    unsigned int edges_hit_count;
    unsigned int edges_miss_count;
} CacheInfo;

#define CACHE_HIT 1
#define CACHE_MISS 2

CacheInfo* init_cache(size_t max_cache_size_images,
                      size_t max_cache_size_tree_nodes,
                      size_t tile_size_bytes);

unsigned char get_tile_data(unsigned int tile_id,
                            CacheInfo *const cache_info,
                            unsigned char **const tile_data);

unsigned char get_diff(unsigned long key,
                       CacheInfo *cache_info, unsigned short *const diff_pixels);

inline static unsigned int calc_images_nodes_cache_size(const CacheInfo* const cache_info) {
    return cache_info->images_nodes_in_cache * sizeof(GenericNode) + cache_info->images_nodes_in_cache * cache_info->tile_size_bytes;
}

inline static unsigned int calc_edge_nodes_cache_size(const CacheInfo* const cache_info) {
    return cache_info->edges_nodes_in_cache * sizeof(GenericNode);
}

void delete_images_tail(CacheInfo *const cache_info);

void delete_edges_tail(CacheInfo *const cache_info);

void push_image_to_cache(unsigned int tile_id,
                         unsigned char *tile_data,
                         CacheInfo* cache_info);

void push_edge_to_cache(unsigned long key,
                        unsigned short int diff_pixels,
                        CacheInfo* cache_info);

void delete_cache(CacheInfo* cache_info);

static void edge_data_destructor(void* data) {
    free(data);
}

static void image_data_destructor(void* data) {
    free(data);
}

#endif // CACHE_UTILS_H
