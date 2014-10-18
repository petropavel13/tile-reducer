#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs_utils.h"
#include <logging.h>
#include <params.h>
#include <tile_utils.h>
#include <reduce_utils.h>

typedef struct file_path_list {
    char* file_path;
    struct file_path_list* next;
} file_path_list;

file_path_list* file_path_list_new(char* const file_path) {
    file_path_list* const fp_list = malloc(sizeof(file_path_list));
    fp_list->file_path = file_path;
    fp_list->next = NULL;

    return fp_list;
}

void file_path_list_free(file_path_list* const head) {
    if (head == NULL) return;

    if (head->file_path != NULL) free(head->file_path);

    file_path_list_free(head->next);
    free(head);
}

typedef struct fs_walker_context {
    unsigned int count;
    file_path_list* files_list;
} fs_walker_context;


void read_files_callback(const char* file_path, void* const context) {
    fs_walker_context* const ctx = context;

    ctx->count++;
    ctx->files_list->file_path = strcpy(malloc(sizeof(char) * strlen(file_path) + 1), file_path);
    ctx->files_list = ctx->files_list->next = file_path_list_new(NULL);
}

typedef struct tile_reduce_context {
    unsigned int reduced_count;
} tile_reduce_context;

void iterate_tree_callback(GenericNode* const node, void* const context) {
    tile_reduce_context* ctx = context;

    const unsigned int left_id = node->key;
    const unsigned int right_id = *((unsigned int*)node->data);

    if (left_id != right_id) {
        ctx->reduced_count++;
        tile_reducer_log_debug("%u reduced by %u", left_id, right_id);
    }
}

int main(int argc, char** argv)
{
    tile_reducer_params arp = tile_reducer_params_make_from_args(argc, argv);

    tile_reducer_log_init();

    file_path_list* const fp_head = file_path_list_new(NULL);

    fs_walker_context fs_walker_ctx;
    fs_walker_ctx.count = 0;
    fs_walker_ctx.files_list = fp_head;

    read_files_in_folder_recursive(arp.path, &fs_walker_ctx, &read_files_callback);

    const unsigned int total = fs_walker_ctx.count;

    tile_reducer_log_info("Total files count %u", total);

    Tile** const tiles = malloc(sizeof(Tile*) * total);

    file_path_list* t_fp = fp_head;

    char* t_path_copy = NULL;

    for (unsigned int i = 0; t_fp->next != NULL; ++i, t_fp = t_fp->next) {
        t_path_copy = malloc(sizeof(char)* strlen(t_fp->file_path) + 1);

        tiles[i] = tile_new(read_tile(t_fp->file_path), i, strcpy(t_path_copy, t_fp->file_path));
    }

    file_path_list_free(fp_head);

    reduce_results_t* const results = reduce_tiles(tiles, total, arp);

    tile_reduce_context ctx;
    ctx.reduced_count = 0;

    iterate_tree(results, &ctx, &iterate_tree_callback);

    tile_reducer_log_info("%u tiles reduced to %u with max different pixels = %u", total, ctx.reduced_count, arp.max_diff_pixels);

    destroy_tree(results, &free);

    for (unsigned int i = 0; i < total; ++i) {
        tile_free(tiles[i]);
    }

    free(tiles);

    return 0;
}

