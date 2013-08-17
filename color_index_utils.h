#ifndef COLOR_INDEX_UTILS_H
#define COLOR_INDEX_UTILS_H

#include "tile_utils.h"
#include "generic_avl_tree.h"
#include "cache_utils.h"
#include "db_utils.h"

typedef struct TileColor {
    unsigned int tile_id;
    unsigned int color;
    unsigned int repeat_count;
} TileColor;

typedef struct TilesTree {
    GenericNode* root_node;
    TreeInfo* tree_info;
} TilesTree;

TilesTree* init_tiles_tree(void);

void index_tile(const Tile* const tile,
                TilesTree *const tiles_tree);


TileColor* create_or_get_tile_color(unsigned int tile_id, unsigned int color, TilesTree* tiles_tree);

void destroy_tile_color_tree(TilesTree* tiles_tree);

void flush_tiles_colors_tree(const TilesTree* const tiles_tree, const DbInfo* const db_info);

void flush_tiles_colors_node(const GenericNode* const tile_color_node, const DbInfo* const db_info);

static void tile_color_destructor(void* data) {
    free(data);
}

static inline unsigned int calc_color(unsigned char* rgba_color) {
    return (((unsigned int) rgba_color[0]) << 24) +
            ((unsigned int) (rgba_color[1]) << 16) +
            ((unsigned int) (rgba_color[2]) << 8) +
            (unsigned int) rgba_color[3];
}

#endif // COLOR_INDEX_UTILS_H
