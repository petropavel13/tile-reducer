#include "color_index_utils.h"


TilesTree* init_tiles_tree(void) {
    TilesTree* const tiles_tree = malloc(sizeof(TilesTree));
    tiles_tree->root_node = NULL;
    tiles_tree->tree_info = malloc(sizeof(TreeInfo));
    tiles_tree->tree_info->data_destructor = &tile_color_destructor;

    return tiles_tree;
}

void index_tile(const Tile* const tile, TilesTree *const tiles_tree) {
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

TileColor* create_or_get_tile_color(unsigned int tile_id, unsigned int color, TilesTree* tiles_tree) {
    const unsigned long key = make_key(tile_id, color);

    GenericNode* const root_node = tiles_tree->root_node;

    if(root_node != NULL) {
        GenericNode* const node = find(root_node, key);

        if(node == NULL) {
            goto create_and_return;
        } else {
            return node->data;
        }
    } else {
        goto create_and_return;
    }

create_and_return:
    {
        TileColor* const new_color = create_tile_color(tile_id, color);

        tiles_tree->root_node = insert(root_node, key, new_color);

        return new_color;
    }
}

TileColor* create_tile_color(unsigned int tile_id, unsigned int color) {
    TileColor* const tile_color = malloc(sizeof(TileColor));
    tile_color->color = color;
    tile_color->tile_id = tile_id;
    tile_color->repeat_count = 0;

    return tile_color;
}

void destroy_tile_color_tree(TilesTree* tiles_tree) {
    destroy_tree(tiles_tree->root_node, tiles_tree->tree_info);
}

void flush_tiles_colors_tree(const TilesTree* const tiles_tree, const DbInfo* const db_info) {
    if(tiles_tree->root_node != NULL) {
        drop_index_tile_color(db_info);

        begin_transaction(db_info);

        flush_tiles_colors_node(tiles_tree->root_node, db_info);

        flush_buffer_tiles_colors(db_info);

        commit_transaction(db_info);

        create_index_tile_color(db_info);
    }
}

void flush_tiles_colors_node(const GenericNode* const tile_color_node, const DbInfo* const db_info) {
    if(tile_color_node != NULL) {
        const TileColor* const tile_color = tile_color_node->data;
        insert_tile_color(tile_color->tile_id, tile_color->color, tile_color->repeat_count, db_info);

        flush_tiles_colors_node(tile_color_node->left, db_info);
        flush_tiles_colors_node(tile_color_node->right, db_info);
    }
}
