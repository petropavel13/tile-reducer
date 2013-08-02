#ifndef CACHE_UTILS_H
#define CACHE_UTILS_H

//#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>

typedef struct CacheNode
{
    struct CacheNode* prev;
    unsigned int tile_id;
    unsigned char* tile_data;
    struct CacheNode* next;
} CacheNode;


typedef struct CacheEdge {
    struct CacheEdge* parent;
    struct CacheEdge* left;
    struct CacheEdge* right;
    unsigned long key;
    unsigned short int diff_pixels;
} CacheEdge;

typedef struct CacheEdgeSequence {
    CacheEdge* edge;
    struct CacheEdgeSequence* prev;
} CacheEdgeSequence;


typedef struct CacheInfo
{
    size_t max_cache_size_images;
    size_t max_cache_size_edges;

    CacheNode* cache_image_head;
    CacheNode* cache_image_tail;

    CacheEdge* cache_edge_root;
    CacheEdgeSequence* cache_edge_head;

    unsigned long min_tree_key;
    unsigned long max_tree_key;

    unsigned int images_in_cache;
    size_t tile_size_bytes;
    unsigned int image_hit_count;
    unsigned int image_miss_count;

    unsigned int edges_in_cache;
    unsigned int edges_hit_count;
    unsigned int edges_miss_count;
} CacheInfo;

#define CACHE_HIT 1
#define CACHE_MISS 2

CacheInfo* init_cache(size_t max_cache_size_images, size_t max_cache_size_edges, size_t tile_size_bytes);

unsigned char get_tile_data(unsigned int tile_id,
                            CacheInfo *const cache_info,
                            unsigned char **const tile_data);

unsigned char get_diff(unsigned int left_tile_id,
                       unsigned int right_tile_id,
                       CacheInfo *cache_info, unsigned short *diff_pixels);

inline static unsigned int calc_images_cache_size(CacheInfo* cache_info) {
    return cache_info->images_in_cache * sizeof(CacheNode) + cache_info->images_in_cache * cache_info->tile_size_bytes;
}

inline static unsigned int calc_edges_cache_size(CacheInfo* cache_info) {
    return cache_info->edges_in_cache * sizeof(CacheEdge);
}

void delete_images_tail(CacheInfo *const cache_info);

void delete_edges_tail(CacheInfo *const cache_info);

void push_image_to_cache(unsigned int tile_id,
                         unsigned char *tile_data,
                         CacheInfo* cache_info);

void push_edge_to_cache(unsigned int left_tile_id,
                        unsigned int right_tile_id,
                        unsigned short int diff_pixels,
                        CacheInfo* cache_info);

void clear_cache(CacheInfo *const cache_info);

static inline void sort_min_max(const unsigned int* min, const unsigned int* max) {
    if(max < min) {
        const unsigned int* tmp = max;
        max = min;
        min = tmp;
    }
}

static inline unsigned long make_key(const unsigned int x, const unsigned int y) {
    const unsigned int max = x;
    const unsigned int min = y;
    sort_min_max(&min, &max);

    return max * max + max + min;
}

CacheEdge* find_edge(CacheEdge *const head, CacheEdge *const fallback_edge, unsigned long key);

void rebalance(CacheEdge* const new_root, CacheEdge* const parent);

#endif // CACHE_UTILS_H
