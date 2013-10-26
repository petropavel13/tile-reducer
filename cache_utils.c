#include "cache_utils.h"

CacheInfo* init_cache(size_t max_cache_size_images, size_t max_cache_size_tree_nodes, size_t tile_size_bytes) {
    CacheInfo* const cache_info = malloc(sizeof(CacheInfo));

    cache_info->max_cache_size_images_nodes = max_cache_size_images;
    cache_info->max_cache_size_edge_nodes = max_cache_size_tree_nodes;
    cache_info->tile_size_bytes = tile_size_bytes;

    cache_info->images_root_node = NULL;

    cache_info->images_tree_info = malloc(sizeof(TreeInfo));
    cache_info->images_tree_info->data_destructor = &image_data_destructor;


    cache_info->edges_root_node = NULL;

    cache_info->edges_tree_info = malloc(sizeof(TreeInfo));
    cache_info->edges_tree_info->data_destructor = &edge_data_destructor;

    cache_info->images_nodes_in_cache = 0;
    cache_info->image_hit_count = 0;
    cache_info->image_miss_count = 0;

    cache_info->edges_nodes_in_cache = 0;
    cache_info->edges_hit_count = 0;
    cache_info->edges_miss_count = 0;

    return cache_info;
}


unsigned char get_tile_data(unsigned int tile_id,
                            CacheInfo* const cache_info,
                            unsigned char** const tile_data) {
    GenericNode* const current_images_root = cache_info->images_root_node;

    if(current_images_root == NULL) {
        cache_info->image_miss_count++;
        return CACHE_MISS;
    }

    const GenericNode* const images_tree_node = find(current_images_root, tile_id);

    if(images_tree_node == NULL) {
        cache_info->image_miss_count++;
        return CACHE_MISS;
    } else {
        (*tile_data) = images_tree_node->data;
        cache_info->image_hit_count++;
        return CACHE_HIT;
    }
}

unsigned char get_diff(unsigned long key,
                       CacheInfo *cache_info,
                       unsigned short* const diff_pixels) {
    GenericNode* const current_edge_root = cache_info->edges_root_node;

    if(current_edge_root == NULL) {
        cache_info->edges_miss_count++;
        return CACHE_MISS;
    }

    const GenericNode* const edges_tree_node = find(current_edge_root, key);

    if(edges_tree_node == NULL) {
        cache_info->edges_miss_count++;
        return CACHE_MISS;
    } else {
        (*diff_pixels) = *((unsigned short int*)edges_tree_node->data);
        cache_info->edges_hit_count++;
        return CACHE_HIT;
    }
}


void delete_images_tail(CacheInfo* const cache_info) {
    cache_info->images_root_node = remove_node(cache_info->images_root_node, find_min(cache_info->images_root_node)->key, cache_info->images_tree_info);

    cache_info->images_nodes_in_cache--;
}

void delete_edges_tail(CacheInfo* const cache_info) {
    cache_info->edges_root_node = remove_node(cache_info->edges_root_node, find_min(cache_info->edges_root_node)->key, cache_info->edges_tree_info);

    cache_info->edges_nodes_in_cache--;
}


void push_image_to_cache(unsigned int tile_id,
                         unsigned char* tile_data,
                         CacheInfo* cache_info) {
    while(calc_images_nodes_cache_size(cache_info) >= cache_info->max_cache_size_images_nodes) {
        delete_images_tail(cache_info);
    }

    GenericNode* const new_tree_head = insert(cache_info->images_root_node, tile_id, tile_data);

    cache_info->images_root_node = new_tree_head;

    cache_info->images_nodes_in_cache++;
}

void push_edge_to_cache(unsigned long key,
                        unsigned short diff_pixels,
                        CacheInfo *cache_info) {
    while(calc_edge_nodes_cache_size(cache_info) >= cache_info->max_cache_size_edge_nodes) {
        delete_edges_tail(cache_info);
    }

    unsigned short int* const l_diff_pixels = malloc(sizeof(unsigned short int));
    (*l_diff_pixels) = diff_pixels;

    GenericNode* const new_tree_head = insert(cache_info->edges_root_node, key, l_diff_pixels);

    cache_info->edges_root_node = new_tree_head;

    cache_info->edges_nodes_in_cache++;
}


void delete_cache(CacheInfo* cache_info) {
    destroy_tree(cache_info->edges_root_node, cache_info->edges_tree_info);
    destroy_tree(cache_info->images_root_node, cache_info->images_tree_info);

    cache_info->image_hit_count = 0;
    cache_info->image_miss_count = 0;

    cache_info->edges_hit_count = 0;
    cache_info->edges_miss_count = 0;

    free(cache_info->images_tree_info);

    free(cache_info->edges_tree_info);

    free(cache_info);
}
