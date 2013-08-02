#include "cache_utils.h"

CacheInfo* init_cache(size_t max_cache_size_images, size_t max_cache_size_edges, size_t tile_size_bytes) {
    CacheInfo* const cache_info = malloc(sizeof(CacheInfo));

    cache_info->max_cache_size_images = max_cache_size_images;
    cache_info->max_cache_size_edges = max_cache_size_edges;
    cache_info->tile_size_bytes = tile_size_bytes;

    cache_info->cache_image_head = NULL;
    cache_info->cache_image_tail = NULL;

    cache_info->cache_edge_root = NULL;
    cache_info->cache_edge_head = NULL;

    cache_info->min_tree_key = UINT64_MAX;
    cache_info->max_tree_key = 0;

    cache_info->images_in_cache = 0;
    cache_info->image_hit_count = 0;
    cache_info->image_miss_count = 0;

    cache_info->edges_in_cache = 0;
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

unsigned char get_diff(unsigned int left_tile_id,
                       unsigned int right_tile_id,
                       CacheInfo* cache_info,
                       unsigned short int* diff_pixels) {
    CacheEdge* const current_edge_root = cache_info->cache_edge_root;

    if(current_edge_root == NULL) {
        return CACHE_MISS;
    }

    const unsigned long key = make_key(left_tile_id, right_tile_id);

    const CacheEdge* const edge = find_edge(current_edge_root, current_edge_root, key);

    if(edge == NULL) {
        return CACHE_MISS;
    }

    if(edge->key == key) {
        (*diff_pixels) = edge->diff_pixels;

        return CACHE_HIT;
    }

    return CACHE_MISS;
}


void delete_images_tail(CacheInfo* const cache_info) {
    CacheNode* const tail = cache_info->cache_image_tail;
    tail->next->prev = NULL;

    cache_info->cache_image_tail = tail->next;

    free(tail->tile_data);
    free(tail);

    cache_info->images_in_cache--;
}

void delete_edges_tail(CacheInfo* const cache_info) {
    CacheEdgeSequence* const new_sequence_head = cache_info->cache_edge_head->prev;
    CacheEdge* const parent_edge = cache_info->cache_edge_head->edge->parent;
    parent_edge->left = NULL;
    parent_edge->right = NULL;

    free(cache_info->cache_edge_head->edge);
    free(cache_info->cache_edge_head);

    cache_info->cache_edge_head = new_sequence_head;

    cache_info->edges_in_cache--;
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

void push_edge_to_cache(unsigned int left_tile_id,
                        unsigned int right_tile_id,
                        unsigned short diff_pixels, CacheInfo *cache_info) {
    while(calc_edges_cache_size(cache_info) >= cache_info->max_cache_size_edges) {
        delete_edges_tail(cache_info);
    }

    const unsigned long key = make_key(left_tile_id, right_tile_id);

    CacheEdge* const similar_edge = find_edge(cache_info->cache_edge_root, cache_info->cache_edge_root, key);

    CacheEdge* const new_edge = malloc(sizeof(CacheEdge));
    new_edge->key = key;
    new_edge->diff_pixels = diff_pixels;
    new_edge->left = NULL;
    new_edge->right = NULL;
    new_edge->parent = NULL;

    if(similar_edge != NULL) {
        if ((key > similar_edge->key)) {

            if(similar_edge->parent != NULL) {

                if(key < similar_edge->parent->key) {
                    similar_edge->parent->left = new_edge;
                    new_edge->parent = similar_edge->parent;
                    new_edge->left = similar_edge;
                } else {
                    similar_edge->parent->right = new_edge;
                    new_edge->parent = similar_edge->parent;
                    new_edge->right = similar_edge;
                }

                similar_edge->parent = new_edge;
            } else {
                similar_edge->right = new_edge;
            }

        } else {
            similar_edge->left = new_edge;
            new_edge->parent = similar_edge;
        }
    } else {
        cache_info->cache_edge_root = new_edge;
    }

    CacheEdgeSequence* const seq_head = cache_info->cache_edge_head;

    CacheEdgeSequence* const new_seq_head = malloc(sizeof(CacheEdgeSequence));

    if(seq_head == NULL) {
        new_seq_head->prev = NULL;
        cache_info->cache_edge_head = new_seq_head;
    } else {
        new_seq_head->prev = seq_head;
    }

    new_seq_head->edge = new_edge;


    if(cache_info->min_tree_key > key) {
        cache_info->min_tree_key = key;
    }

    if(cache_info->max_tree_key < key) {
        cache_info->max_tree_key = key;
    }

    CacheEdge* const current_cache_root = cache_info->cache_edge_root;

    const unsigned long max = cache_info->max_tree_key;
    const unsigned long min = cache_info->min_tree_key;

    CacheEdge* const middle = find_edge(current_cache_root, current_cache_root,  min + ((max - min) / 2));

    if(current_cache_root != middle) {
        rebalance(middle, NULL);
        cache_info->cache_edge_root = middle; // rebalance
    }

    cache_info->edges_in_cache++;
}


void clear_cache(CacheInfo* const cache_info) {
    while(cache_info->images_in_cache > 0) {
        delete_images_tail(cache_info);
    }

    while(cache_info->edges_in_cache > 0) {
        delete_edges_tail(cache_info);
    }

    cache_info->image_hit_count = 0;
    cache_info->image_miss_count = 0;

    cache_info->edges_hit_count = 0;
    cache_info->edges_miss_count = 0;

    cache_info->cache_image_head = NULL;
    cache_info->cache_image_tail = NULL;

    cache_info->cache_edge_root = NULL;
}

CacheEdge* find_edge(CacheEdge* const head, CacheEdge* const fallback_edge, unsigned long key) {
    if(head == NULL)
        return fallback_edge;

    if(head->key > key) {
        return find_edge(head->left, head, key);
    } else if(head->key < key) {
        return find_edge(head->right, head, key);
    } else {
        return head;
    }
}

void rebalance(CacheEdge* const new_root, CacheEdge* const parent) {
    if(new_root->parent != NULL) {
        const unsigned long parent_key = new_root->parent->key;

        if(parent_key > new_root->key) {
            new_root->right = new_root->parent;
            rebalance(new_root->right, new_root);
        } else {
            new_root->left = new_root->parent;
            rebalance(new_root->left, new_root);
        }
    } else { // past root
        if(parent->key > new_root->key) {
            new_root->right = NULL;
        } else {
            new_root->left = NULL;
        }
    }

    new_root->parent = parent;
}
