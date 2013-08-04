#include "cache_utils.h"

CacheInfo* init_cache(size_t max_cache_size_images, size_t max_cache_size_tree_nodes, size_t tile_size_bytes) {
    CacheInfo* const cache_info = malloc(sizeof(CacheInfo));

    cache_info->max_cache_size_images = max_cache_size_images;
    cache_info->max_cache_size_tree_nodes = max_cache_size_tree_nodes;
    cache_info->tile_size_bytes = tile_size_bytes;

    cache_info->cache_image_head = NULL;
    cache_info->cache_image_tail = NULL;

    cache_info->root_tree_node = NULL;

    cache_info->images_in_cache = 0;
    cache_info->image_hit_count = 0;
    cache_info->image_miss_count = 0;

    cache_info->tree_nodes_in_cache = 0;
    cache_info->edges_hit_count = 0;
    cache_info->edges_miss_count = 0;

    return cache_info;
}


unsigned char get_tile_data(unsigned int tile_id,
                            CacheInfo* const cache_info,
                            unsigned char** const tile_data) {
    CacheNode* const current_image_head = cache_info->cache_image_head;


    if(current_image_head == NULL) {
        return CACHE_MISS;
    }

    if(current_image_head->tile_id == tile_id) {
        (*tile_data) = current_image_head->tile_data;

        cache_info->image_hit_count++;

        return CACHE_HIT;
    }

    CacheNode* temp = current_image_head;

    while((temp = temp->prev) != NULL) {
        if(temp->tile_id == tile_id) {
            (*tile_data) = temp->tile_data;

            CacheNode* const new_cache_head = temp;

            // move hit node to head

            CacheNode* const node_before_hit = new_cache_head->prev;
            CacheNode* const node_after_hit = new_cache_head->next;

            if(node_before_hit != NULL) {
                node_before_hit->next = node_after_hit;
            } else {
                cache_info->cache_image_tail = node_after_hit;
            }

            node_after_hit->prev = node_before_hit; // NULL or node_before_hit

            new_cache_head->prev = current_image_head;
            new_cache_head->next = NULL;
            current_image_head->next = new_cache_head;

            cache_info->cache_image_head = new_cache_head;

            cache_info->image_hit_count++;

            return CACHE_HIT;
        }
    }

    cache_info->image_miss_count++;

    return CACHE_MISS;
}

unsigned char get_diff(unsigned long key,
                       CacheInfo *cache_info,
                       unsigned short *diff_pixels) {
    TreeNode* const current_edge_root = cache_info->root_tree_node;

    if(current_edge_root == NULL) {
        cache_info->edges_miss_count++;
        return CACHE_MISS;
    }

    const TreeNode* const tree_node = find(current_edge_root, key);

    if(tree_node == NULL) {
        cache_info->edges_miss_count++;
        return CACHE_MISS;
    } else {
        (*diff_pixels) = tree_node->diff_pixels;
        cache_info->edges_hit_count++;
        return CACHE_HIT;
    }
}


void delete_images_tail(CacheInfo* const cache_info) {
    CacheNode* const tail = cache_info->cache_image_tail;
    tail->next->prev = NULL;

    cache_info->cache_image_tail = tail->next;

    free(tail->tile_data);
    free(tail);

    cache_info->images_in_cache--;
}

void delete_tree_tail(CacheInfo* const cache_info) {
    cache_info->root_tree_node = remove_min(cache_info->root_tree_node);

    cache_info->tree_nodes_in_cache--;
}


void push_image_to_cache(unsigned int tile_id,
                         unsigned char* tile_data,
                         CacheInfo* cache_info) {
    while(calc_images_cache_size(cache_info) >= cache_info->max_cache_size_images) {
        delete_images_tail(cache_info);
    }

    CacheNode* const old_cache_head = cache_info->cache_image_head;

    CacheNode* const new_cache_head = malloc(sizeof(CacheNode));

    new_cache_head->prev = old_cache_head;
    new_cache_head->tile_id = tile_id;
    new_cache_head->tile_data = tile_data;
    new_cache_head->next = NULL;

    if(old_cache_head != NULL) {
        old_cache_head->next = new_cache_head;
    } else { // it's first push - > head == tail
        cache_info->cache_image_tail = new_cache_head;
    }

    cache_info->cache_image_head = new_cache_head;
    cache_info->images_in_cache++;
}

void push_edge_to_cache(unsigned long key,
                        unsigned short diff_pixels,
                        CacheInfo *cache_info) {
    while(calc_tree_nodes_cache_size(cache_info) >= cache_info->max_cache_size_tree_nodes) {
        delete_tree_tail(cache_info);
    }

    TreeNode* const new_tree_head = insert(cache_info->root_tree_node, key, diff_pixels);

    cache_info->root_tree_node = new_tree_head;

    cache_info->tree_nodes_in_cache++;
}


void clear_cache(CacheInfo* const cache_info) {
    while(cache_info->images_in_cache > 0) {
        delete_images_tail(cache_info);
    }

    while(cache_info->tree_nodes_in_cache > 0) {
        delete_tree_tail(cache_info);
    }

    cache_info->image_hit_count = 0;
    cache_info->image_miss_count = 0;

    cache_info->edges_hit_count = 0;
    cache_info->edges_miss_count = 0;

    cache_info->cache_image_head = NULL;
    cache_info->cache_image_tail = NULL;

    cache_info->root_tree_node = NULL;
}
