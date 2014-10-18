#ifndef CACHE_UTILS_H
#define CACHE_UTILS_H

#include "generic_avl_tree.h"

typedef struct CacheInfo {
    size_t max_cache_size_images_nodes;
    size_t max_cache_size_edge_nodes;

    size_t tile_size_bytes;

    GenericNode* images_root_node;

    GenericNode* diffs_root_node;

    unsigned long images_nodes_in_cache;
    unsigned long image_hit_count;
    unsigned long image_miss_count;

    unsigned long diff_nodes_in_cache;
    unsigned long diffs_hit_count;
    unsigned long diffs_miss_count;
} CacheInfo;


typedef enum CacheSearchResult {
    CACHE_HIT,
    CACHE_MISS
} CacheSearchResult;


CacheInfo* cache_info_new(const size_t max_cache_size_images,
                          const size_t max_cache_size_tree_nodes,
                          const size_t tile_size_bytes);

static inline void cache_info_free(CacheInfo* const cache_info) {
    destroy_tree(cache_info->diffs_root_node, &free);
    destroy_tree(cache_info->images_root_node, &free);

    free(cache_info);
}


CacheSearchResult get_tile_data(const unsigned int tile_id,
                                CacheInfo* const cache_info,
                                unsigned char** const tile_data);

CacheSearchResult get_diff_from_cache(unsigned long key,
                                      CacheInfo *const cache_info,
                                      unsigned int* const diff_pixels);

void push_image_to_cache(const unsigned int tile_id,
                         unsigned char* const tile_data,
                         CacheInfo* const cache_info);

void delete_image_in_cache(const unsigned int tile_id,
                           CacheInfo* const cache_info);

void push_diff_to_cache(const unsigned long key,
                        const unsigned int diff_pixels,
                        CacheInfo* const cache_info);

#endif // CACHE_UTILS_H
