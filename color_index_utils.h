#ifndef COLOR_INDEX_UTILS_H
#define COLOR_INDEX_UTILS_H

#include "tile_utils.h"
#include "generic_avl_tree.h"
#include "db_utils.h"

typedef struct TileColor {
    unsigned int tile_id;
    unsigned int color;
    unsigned int repeat_count;
} TileColor;

typedef struct TilesColorsTree {
    GenericNode* root_node;
    unsigned long colors_count;

    unsigned long tiles_count;

    void(*index_callback)(unsigned char);
    unsigned long current_index;
    unsigned char last_index_persent;

    void(*flush_callback)(unsigned char);
    unsigned long current_flush;
    unsigned char last_flush_persent;

    DbInfo* db_info;
} TilesColorsTree;

typedef struct TCTParams {
    GenericNode* tiles_tree;
    void(*index_callback)(unsigned char);
    void(*flush_callback)(unsigned char);

    DbInfo* db_info;
} TCTParams;


TCTParams make_tct_params(GenericNode * const tiles_tree,
                          void(*index_callback)(unsigned char),
                          void(*flush_callback)(unsigned char),
                          DbInfo* const db_info);

TilesColorsTree* create_tiles_colors_tree(const GenericNode* const tiles_tree,
                                          DbInfo * const db_info,
                                          void(*index_callback)(unsigned char),
                                          void(*flush_callback)(unsigned char));

void* index_tree_and_flush_result(void* arg);
void index_tree(TilesColorsTree* const tiles_colors_tree, const GenericNode* const tiles_tree);

void index_tile(TilesColorsTree *const tiles_tree, const Tile* const tile);


TileColor* create_or_get_tile_color(const unsigned int tile_id,
                                    const unsigned int color,
                                    TilesColorsTree * const tiles_tree);
TileColor* create_tile_color(const unsigned int tile_id, const unsigned int color);

void destroy_tile_color_tree(TilesColorsTree* tiles_tree);

void flush_tiles_colors_tree(TilesColorsTree *const tiles_colors_tree);

void flush_tiles_colors_node(const GenericNode* const tile_color_node, TilesColorsTree * const tiles_colors_tree);

void tile_color_destructor(void* data);

static inline unsigned int calc_color(unsigned char* rgba_color) {
    return (((unsigned int) rgba_color[0]) << 24) +
           (((unsigned int) rgba_color[1]) << 16) +
           (((unsigned int) rgba_color[2]) << 8) +
           (((unsigned int) rgba_color[3] << 0));
}

#endif // COLOR_INDEX_UTILS_H
