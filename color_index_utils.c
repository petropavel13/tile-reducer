#include "color_index_utils.h"


TCTParams make_tct_params(GenericNode* const tiles_tree,
                          void(*index_callback)(unsigned char),
                          void(*flush_callback)(unsigned char),
                          DbInfo * const db_info) {
    TCTParams params = { tiles_tree, index_callback, flush_callback, db_info };

    return params;
}

TilesColorsTree* create_tiles_colors_tree(const GenericNode* const tiles_tree,
                                          DbInfo * const db_info,
                                          void(*index_callback)(unsigned char),
                                          void(*flush_callback)(unsigned char)) {
    TilesColorsTree* const tiles_colors_tree = malloc(sizeof(TilesColorsTree));
    tiles_colors_tree->root_node = create_node(0, NULL);
    tiles_colors_tree->colors_count = 0;

    tiles_colors_tree->db_info = db_info;

    tiles_colors_tree->tiles_count = 0;

    calc_elements_count(tiles_tree, &tiles_colors_tree->tiles_count);

    tiles_colors_tree->index_callback = index_callback;
    tiles_colors_tree->current_index = 0;
    tiles_colors_tree->last_index_persent = 0;

    tiles_colors_tree->flush_callback = flush_callback;
    tiles_colors_tree->current_flush = 0;
    tiles_colors_tree->last_flush_persent = 0;

    return tiles_colors_tree;
}

void* index_tree_and_flush_result(void *arg) {
    TCTParams* const params = (TCTParams*)arg;

    TilesColorsTree* const tc_tree = create_tiles_colors_tree(params->tiles_tree,
                                                                 params->db_info,
                                                                 params->index_callback,
                                                                 params->flush_callback);

    index_tree(tc_tree, params->tiles_tree);

    if(tc_tree->index_callback != NULL) {
        tc_tree->index_callback(146);
    }

    flush_tiles_colors_tree(tc_tree);

    if(tc_tree->flush_callback != NULL) {
        tc_tree->flush_callback(146);
    }

    destroy_tile_color_tree(tc_tree);

    return NULL;
}

void index_tree(TilesColorsTree* const tiles_colors_tree, const GenericNode* const tiles_tree) {
    if(tiles_tree != NULL) {
        index_tile(tiles_colors_tree, (const Tile* const)tiles_tree->data);

        if(tiles_colors_tree->index_callback != NULL) {
            const unsigned char current_percent = (++tiles_colors_tree->current_index / (tiles_colors_tree->tiles_count / 100.0));

            if(current_percent != tiles_colors_tree->last_index_persent) {
                tiles_colors_tree->index_callback(tiles_colors_tree->last_index_persent = current_percent);
            }
        }

        index_tree(tiles_colors_tree, tiles_tree->left);
        index_tree(tiles_colors_tree, tiles_tree->right);
    }
}

void index_tile(TilesColorsTree *const tiles_tree, const Tile* const tile) {
    unsigned char* raw_image = NULL;

    get_tile_pixels(tile->tile_file, &raw_image);

    unsigned int temp_color = 0;
    TileColor* temp_tile_color = NULL;

    for (unsigned int i = 0; i < TILE_SIZE; i += 4) {
        temp_color = calc_color(&raw_image[i]);

        temp_tile_color = create_or_get_tile_color(tile->tile_id, temp_color, tiles_tree);

        temp_tile_color->repeat_count++;
    }

    free(raw_image);
}

TileColor* create_or_get_tile_color(const unsigned int tile_id,
                                    const unsigned int color,
                                    TilesColorsTree* const tiles_tree) {
    const unsigned long key = make_key(tile_id, color);

    GenericNode* const root_node = tiles_tree->root_node;

    GenericNode* const node = find(root_node, key);

    if(node == NULL) {
        TileColor* const new_color = create_tile_color(tile_id, color);

        tiles_tree->root_node = insert(root_node, key, new_color);

        tiles_tree->colors_count++;

        return new_color;
    }

    return node->data;
}

TileColor* create_tile_color(const unsigned int tile_id, const unsigned int color) {
    TileColor* const tile_color = malloc(sizeof(TileColor));
    tile_color->color = color;
    tile_color->tile_id = tile_id;
    tile_color->repeat_count = 0;

    return tile_color;
}

void destroy_tile_color_tree(TilesColorsTree* tiles_tree) {
    destroy_tree(tiles_tree->root_node, &tile_color_destructor);
    free(tiles_tree);
}

void flush_tiles_colors_tree(TilesColorsTree* const tiles_colors_tree) {
    tiles_colors_tree->root_node = remove_node(tiles_colors_tree->root_node, 0, &tile_color_destructor);

    drop_index_tile_color(tiles_colors_tree->db_info);

    flush_tiles_colors_node(tiles_colors_tree->root_node, tiles_colors_tree);

    flush_db_buffer(tiles_colors_tree->db_info);

    create_index_tile_color(tiles_colors_tree->db_info);
}

void flush_tiles_colors_node(const GenericNode* const tile_color_node, TilesColorsTree* const tiles_colors_tree) {
    if(tile_color_node != NULL) {
        const TileColor* const tile_color = tile_color_node->data;
        insert_tile_color_using_buffer(tile_color->tile_id, tile_color->color, tile_color->repeat_count, tiles_colors_tree->db_info);

        if(tiles_colors_tree->flush_callback!= NULL) {
            const unsigned char current_percent = (++tiles_colors_tree->current_flush / (tiles_colors_tree->colors_count / 100.0));

            if(tiles_colors_tree->last_flush_persent != current_percent) {
                tiles_colors_tree->flush_callback(tiles_colors_tree->last_flush_persent = current_percent);
            }
        }

        flush_tiles_colors_node(tile_color_node->left, tiles_colors_tree);
        flush_tiles_colors_node(tile_color_node->right, tiles_colors_tree);
    }
}

void tile_color_destructor(void* data) {
    free(data);
}
