#include "cache_utils.h"

#include <string.h> // for memcpy

static inline size_t images_size_in_cache(const CacheInfo* const cache_info) {
    return cache_info->images_nodes_in_cache * sizeof(GenericNode) + cache_info->images_nodes_in_cache * cache_info->tile_size_bytes;
}

static inline size_t diffs_size_in_cache(const CacheInfo* const cache_info) {
    return cache_info->diff_nodes_in_cache * sizeof(GenericNode) + cache_info->diff_nodes_in_cache * sizeof(unsigned int);
}

inline CacheInfo* cache_info_new(const size_t max_cache_size_images,
                                 const size_t max_cache_size_tree_nodes,
                                 const size_t tile_size_bytes) {
    CacheInfo* const cache_info = malloc(sizeof(CacheInfo));

    cache_info->max_cache_size_images_nodes = max_cache_size_images;
    cache_info->max_cache_size_edge_nodes = max_cache_size_tree_nodes;
    cache_info->tile_size_bytes = tile_size_bytes;

    cache_info->images_root_node = NULL;

    cache_info->diffs_root_node = NULL;

    cache_info->images_nodes_in_cache = 0;
    cache_info->image_hit_count = 0;
    cache_info->image_miss_count = 0;

    cache_info->diff_nodes_in_cache = 0;
    cache_info->diffs_hit_count = 0;
    cache_info->diffs_miss_count = 0;

    return cache_info;
}


CacheSearchResult get_tile_data(const unsigned int tile_id,
                                CacheInfo* const cache_info,
                                unsigned char** const tile_data) {
    const GenericNode* const images_tree_node = find(cache_info->images_root_node, tile_id);

    if(images_tree_node == NULL) {
        cache_info->image_miss_count++;
        return CACHE_MISS;
    } else {
        (*tile_data) = images_tree_node->data;
        cache_info->image_hit_count++;
        return CACHE_HIT;
    }
}

CacheSearchResult get_diff_from_cache(unsigned long key,
                                      CacheInfo *const cache_info,
                                      unsigned int* const diff_pixels) {
    const GenericNode* const edges_tree_node = find(cache_info->diffs_root_node, key);

    if(edges_tree_node == NULL) {
        cache_info->diffs_miss_count++;
        return CACHE_MISS;
    } else {
        (*diff_pixels) = *((unsigned int*)edges_tree_node->data);
        cache_info->diffs_hit_count++;
        return CACHE_HIT;
    }
}

static inline void delete_images_tail(CacheInfo* const cache_info) {
    cache_info->images_root_node = remove_node(cache_info->images_root_node, find_min(cache_info->images_root_node)->key, &free);

    cache_info->images_nodes_in_cache--;
}

static inline void delete_diffs_tail(CacheInfo* const cache_info) {
    cache_info->diffs_root_node = remove_node(cache_info->diffs_root_node, find_min(cache_info->diffs_root_node)->key, &free);

    cache_info->diff_nodes_in_cache--;
}


void push_image_to_cache(const unsigned int tile_id,
                         unsigned char* const tile_data,
                         CacheInfo *const cache_info) {
    while(images_size_in_cache(cache_info) >= cache_info->max_cache_size_images_nodes) {
        delete_images_tail(cache_info);
    }

    cache_info->images_root_node = insert(cache_info->images_root_node, tile_id, tile_data);

    cache_info->images_nodes_in_cache++;
}

inline void delete_image_in_cache(const unsigned int tile_id,
                                  CacheInfo* const cache_info) {
    const GenericNode* const node_for_delete = find(cache_info->images_root_node, tile_id);

    if (node_for_delete != NULL) {
        cache_info->images_root_node = remove_node(cache_info->images_root_node, tile_id, &free);
        cache_info->images_nodes_in_cache--;
    }
}

void push_diff_to_cache(const unsigned long key,
                        const unsigned int diff_pixels,
                        CacheInfo* const cache_info) {
    while(diffs_size_in_cache(cache_info) >= cache_info->max_cache_size_edge_nodes) {
        delete_diffs_tail(cache_info);
    }

    unsigned int* const l_diff_pixels = malloc(sizeof(unsigned int));
    (*l_diff_pixels) = diff_pixels;

    cache_info->diffs_root_node = insert(cache_info->diffs_root_node, key, l_diff_pixels);

    cache_info->diff_nodes_in_cache++;
}
