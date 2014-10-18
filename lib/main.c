#include <stdio.h>

#include <math.h>

#include "tile_utils.h"
#include "cache_utils.h"
#include "db_utils.h"
#include "cluster_utils.h"
#include "color_index_utils.h"
#include "params.h"
#include "fs_utils.h"
#include "logging.h"

#include <libpq-fe.h>
#include <pthread.h>
#include <log4c.h>

#define IS_STRINGS_EQUAL(str0, st1) (strcmp(str0, st1) == 0)
#define IS_STRINGS_NOT_EQUAL(str0, st1) (strcmp(str0, st1) != 0)

#define PARAM_NOT_FOUND "false"
#define PARAM_FOUND "true"

#define IS_PARAM_FOUND(param) (IS_STRINGS_EQUAL(param, PARAM_NOT_FOUND) == 0)
#define IS_PARAM_NOT_FOUND(param) IS_STRINGS_EQUAL(param, PARAM_NOT_FOUND)

#define CONNECTION_STRING "dbname=tiles_db hostaddr=192.168.0.108 user=postgres port=5432 password=123"
//#define CONNECTION_STRING "dbname=tiles_test hostaddr=172.18.36.131 user=postgres port=5432 password=123"
//#define CONNECTION_STRING "dbname=tiles_db_256 host=/var/run/postgresql user=postgres password=123"


#define LEFT 0
#define RIGHT 1
//#define DEBUG 1

typedef struct FilePathList {
    char* file_path;
    struct FilePathList* next;
} FilePathList;

typedef FilePathList* FilePathListRef;

typedef struct FSWalkerContext {
    unsigned int count;
    FilePathList* files_list;
} FSWalkerContext;

void file_path_list_destructor(FilePathList* const head) {
    if (head == NULL) return;

    if (head->file_path != NULL) free(head->file_path);

    file_path_list_destructor(head->next);
    free(head);
}

void read_files_callback(const char* file_path, void* const context) {
    FSWalkerContext* const ctx = context;

    ctx->count++;
    ctx->files_list->file_path = strcpy(malloc(sizeof(char) * strlen(file_path) + 1), file_path);
    FilePathList* next = ctx->files_list->next = malloc(sizeof(FilePathList));
    next->next = NULL;
    ctx->files_list = next;
}


typedef struct TilesInsertContext {
    unsigned int count;
    unsigned int current_index;
    unsigned int* tiles_ids;
#ifndef NO_LOG
    unsigned char last_log_percent;
#endif
} TilesInsertContext;

TilesInsertContext* tiles_insert_context_new(const unsigned int tiles_count) {
    TilesInsertContext* const ctx = malloc(sizeof(TilesInsertContext));
    ctx->count = tiles_count;
    ctx->current_index = 0;
    ctx->tiles_ids = malloc(sizeof(unsigned int) * tiles_count);

    return ctx;
}

void tiles_insert_context_free(TilesInsertContext* const ctx) {
    free(ctx->tiles_ids);
    free(ctx);
}

void tiles_insert_callback(const unsigned int tile_id, void* const callback_context) {
    TilesInsertContext* const ctx = callback_context;
    ctx->tiles_ids[ctx->current_index++] = tile_id;

#ifndef NO_LOG
    const unsigned char percent_done = ceil(ctx->current_index / (ctx->count / 100.0));

    if (percent_done != ctx->last_log_percent) {
        ctx->last_log_percent = percent_done;

        if (percent_done % 25 == 0) { // 0, 25, 50, 75, 100
            tile_reducer_log_info("Inserting tiles in db: %d%% done", percent_done);
        }
    }

#endif
}

void print_progress_index_colors(const unsigned char percent_done) {
    if(percent_done <= 100) {
        printf("\r                            ");
        printf("\rIndexing tiles colors...%d%%", percent_done);
        fflush(stdout);
    } else {
        printf("\r                                                ");
        printf("\rIndexing tiles colors...done\n");
        fflush(stdout);
    }
}

void print_progress_tiles_colors_db_write(const unsigned char percent_done) {
    if(percent_done <= 100) {
        printf("\r                            ");
        printf("\rWriting tiles colors to db...%d%%", percent_done);
        fflush(stdout);
    } else {
        printf("\r                                                ");
        printf("\rWriting tiles colors to db...done\n");
        fflush(stdout);
    }
}

void print_make_persistent_groups(const unsigned int current, const unsigned int candidates_count) {
    printf("\r                                                ");
    printf("\rMaking persistent groups...%d(%d)", current, candidates_count);
    fflush(stdout);
}


GenericNode* get_head(const GenericNode* const tiles_node, const unsigned char current_level, const unsigned char path) {
    const unsigned char direction = ((unsigned char)((path >> current_level) << 7)) >> 7;

    GenericNode* const next_tree_node = direction == LEFT ? tiles_node->left : tiles_node->right;

    return current_level == 0 ? next_tree_node : get_head(next_tree_node, current_level - 1, path);
}


GenericNode* get_tails(GenericNode* const tail, const GenericNode* const tiles_node, const unsigned char current_level) {
    if (current_level == 0)
        return tail;

    GenericNode* const local_tail = get_tails(insert(tail, ((Tile*)tiles_node->left->data)->tile_id, tiles_node->left->data), tiles_node->left, current_level - 1);

    return get_tails(insert(local_tail, tiles_node->right->key, tiles_node->right->data), tiles_node->right, current_level - 1);
}


void run_index_threads(GenericNode* const tiles_tree, tile_reducer_params arp) {
    const unsigned int num_threads = arp.max_num_threads;

    pthread_t threads[num_threads];
    GenericNode * heads[num_threads];
    DbInfo * connections[num_threads];
    TCTParams th_params[num_threads];

    for (unsigned char i = 0; i < num_threads; ++i) {
        heads[i] = get_head(tiles_tree, num_threads >> 2, i);
        connections[i] = create_db_info(PQconnectdb(CONNECTION_STRING), arp.max_sql_string_size / num_threads);
        th_params[i].tiles_tree = heads[i];
        th_params[i].index_callback = &print_progress_index_colors;
        th_params[i].flush_callback = &print_progress_tiles_colors_db_write;
        th_params[i].db_info = connections[i];
        pthread_create(&threads[i], NULL, &index_tree_and_flush_result, &th_params[i]);
    }


    GenericNode* const tails = get_tails(create_node(tiles_tree->key, tiles_tree->data), tiles_tree, num_threads >> 2);

    DbInfo* const db_info = create_db_info(PQconnectdb(CONNECTION_STRING), arp.max_sql_string_size / num_threads);
    TCTParams params;
    params.tiles_tree = tails;
    params.index_callback = NULL;
    params.flush_callback = NULL;
    params.db_info = db_info;

    index_tree_and_flush_result(&params);

    destroy_tree(tails, NULL);
    destroy_db_info(db_info);

    for (unsigned char i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
        destroy_db_info(connections[i]);
    }
}

char* get_arg(const int argc, char** argv, const char* const key) {
    int i = 0;

    for(; i < argc; ++i) {
        if (strstr(argv[i], key) != NULL) {
            if (strchr(argv[i], '=') != NULL) {
                return &(argv[i][strlen(key) + 1]);
            } else {
                return PARAM_FOUND;
            }
        }
    }

    return PARAM_NOT_FOUND;
}

void print_help() {
    printf("template: ./comparer --path=/path/to/tiles/folder/ --max_diff_pixels=0...65535 [--max_mb_cache=] [--max_num_theads=2^x -> (2,4,8,16,...)]\n");
    printf("example: ./comparer --path=/tiles/opt_easy/ --max_diff_pixels=16 --max_mb_cache=4096 --max_threads=8\n");
    printf("example: ./comparer --path=/tiles/opt_easy/ --max_diff_pixels=64\n");
}


int main(int argc, char** argv)
{
    tile_reducer_params arp = tile_reducer_params_make_default();

    const char* const path = get_arg(argc, argv, "--path");
    const char* const max_diff_pixels_param = get_arg(argc, argv, "--max_diff_pixels");

    if (IS_PARAM_NOT_FOUND(path) || IS_PARAM_NOT_FOUND(max_diff_pixels_param)) {
        print_help();
        return 1;
    } else {
        arp.max_diff_pixels = atoi(max_diff_pixels_param);
    }

    const char* const max_cache_size_param = get_arg(argc, argv, "--max_mb_cache");

    if (IS_PARAM_FOUND(max_cache_size_param)) {
        arp.max_cache_size = ((size_t) atoi(max_cache_size_param)) * 1024 * 1024;
    }

    const char* const max_num_threads_param = get_arg(argc, argv, "--max_threads");

    log4c_init();

    if (IS_PARAM_FOUND(max_num_threads_param)) {
        const unsigned char max_num_threads = atoi(max_num_threads_param);

        arp.max_num_threads = max_num_threads;

        if ((max_num_threads & (~max_num_threads + 1)) != max_num_threads) {
            printf("max_threads must be power of 2!\n\n");
            print_help();
            return 1;
        }
    }

    printf("Tiles folder: \"%s\"\nMax diff. pixels: %u\nCache size: %u MB\nMax threads: %u\n\n",
                          path,
                          arp.max_diff_pixels,
                          (unsigned int) (arp.max_cache_size / 1024 / 1024),
                          (unsigned int) arp.max_num_threads
        );

    FilePathList* const fp_head = malloc(sizeof(FilePathList));

    FSWalkerContext fs_walker_ctx;
    fs_walker_ctx.count = 0;
    fs_walker_ctx.files_list = fp_head;

    read_files_in_folder_recursive(path, &fs_walker_ctx, &read_files_callback);

    const unsigned int total = fs_walker_ctx.count;

    char** const tiles_paths = malloc(sizeof(char*) * total);

    FilePathList* t_fp = fp_head;

    for (unsigned int i = 0; t_fp->next != NULL; ++i, t_fp = t_fp->next) {
        tiles_paths[i] = t_fp->file_path;
    }

    printf("\rTotal tiles count: %d         \n", total);
    fflush(stdout);

    PGconn* const conn = PQconnectdb(CONNECTION_STRING);

    printf("Connecting to db...");
    fflush(stdout);

    if (PQstatus(conn) == CONNECTION_BAD) {
        printf("%s\n", PQerrorMessage(conn));

        return 1;
    }

    DbInfo* const db_info = create_db_info(conn, arp.max_sql_string_size);

    printf("done\n");
    fflush(stdout);    

//    read_tiles_paths(path, tiles_paths, &total, &zero, &percent, &print_progress_paths_read);

    printf("\r                                                ");
    printf("\rReading tiles paths...done\n");
    fflush(stdout);

    create_tables_if_not_exists(db_info);

    clear_all_data(db_info); // DEBUG

    const unsigned int res = check_tiles_in_db(db_info, total);

    unsigned int* pg_ids = NULL;

    unsigned int not_used = 0;

    if(res == TILES_ALREADY_EXISTS) {
        tile_reducer_log_info("Tiles already in db. Reading ids...");

        clear_session_data(db_info);

        read_tiles_ids(db_info, &pg_ids, &not_used);
    } else if(res == NO_TILES_IN_DB) {
        clear_all_data(db_info); // reset sequences

        pg_ids = malloc(sizeof(unsigned int) * total);
        TilesInsertContext* const ctx = tiles_insert_context_new(total);

        insert_tiles_info(db_info, (const char * const * const)tiles_paths, total, ctx, &tiles_insert_callback);

        memcpy(pg_ids, ctx->tiles_ids, sizeof(unsigned int) * total);

        tiles_insert_context_free(ctx);
    } else if(res == TILES_COUNT_MISMATCH) {
        tile_reducer_log_info("Tiles count mismatch. Clean up db.");

        clear_all_data(db_info);

        pg_ids = malloc(sizeof(unsigned int) * total);
        TilesInsertContext* const ctx = tiles_insert_context_new(total);

        insert_tiles_info(db_info, (const char * const * const)tiles_paths, total, ctx, &tiles_insert_callback);

        memcpy(pg_ids, ctx->tiles_ids, sizeof(unsigned int) * total);

        tiles_insert_context_free(ctx);
    }

    const size_t diffs_cache_size = floor(arp.max_cache_size * 0.05);

    CacheInfo* const cache_info = cache_info_new(diffs_cache_size, arp.max_cache_size - diffs_cache_size, TILE_SIZE_BYTES);

    Tile* temp_tile = malloc(sizeof(Tile));
    temp_tile->tile_id = pg_ids[0];
    temp_tile->tile_file = read_tile(tiles_paths[0]);

    GenericNode* all_tiles = create_node(temp_tile->tile_id, temp_tile);

    unsigned int current = 0;
    unsigned char last_percent = 0;

    for (unsigned int i = 1; i < total; ++i)
    {
        temp_tile = malloc(sizeof(Tile));
        temp_tile->tile_id = pg_ids[i];
        temp_tile->tile_file = read_tile(tiles_paths[i]);

        all_tiles = insert(all_tiles, temp_tile->tile_id, temp_tile);

        const unsigned char current_percent = (++current / (total / 100));

        if(last_percent != current_percent) {
            printf("\r                                                ");
            printf("\rLoading tiles into RAM...%d%%", current_percent);
            fflush(stdout);
            last_percent = current_percent;
        }
    }

    printf("\r                                                ");
    printf("\rLoading tiles into RAM...done\n");
    fflush(stdout);

    file_path_list_destructor(fp_head);

    free(tiles_paths);

    if(res != TILES_ALREADY_EXISTS) {
        drop_index_tile_color(db_info);

        if (arp.max_num_threads < 2) {
            TCTParams params;
            params.tiles_tree = all_tiles;
            params.index_callback = &print_progress_index_colors;
            params.flush_callback = &print_progress_tiles_colors_db_write;
            params.db_info = db_info;

            index_tree_and_flush_result(&params);
        } else {
//            printf("Indexing and writing tiles color %u threads...", max_num_threads);
            fflush(stdout);
            run_index_threads(all_tiles, arp);
//            printf("\rIndexing and writing tiles color %u threads...done\n", max_num_threads);
            fflush(stdout);
        }

        create_index_tile_color(db_info);

        printf("\r                                                ");
        printf("\rMaterializing count equality view...");
        fflush(stdout);

        materialize_count_equality(db_info);

        printf("\r                                                ");
        printf("\rMaterializing count equality view...done\n");
        fflush(stdout);

        make_persistent_groups(db_info, all_tiles, total, arp, cache_info, &print_make_persistent_groups);

        printf("\r                                                ");
        printf("\rMaking persistent groups...done\n");
        fflush(stdout);
    }

    if (arp.max_diff_pixels > 0) {
        clusterize_simple(all_tiles, total, arp, db_info, cache_info);
    }

    destroy_tree(all_tiles, &tile_free);

    const unsigned long images_hits = cache_info->image_hit_count;
    const unsigned long images_misses = cache_info->image_miss_count;

    const unsigned long diffs_hits = cache_info->edges_hit_count;
    const unsigned long diffs_misses = cache_info->edges_miss_count;

    printf("images | hits: %lu, misses: %lu, ratio: %.2Lf\n", images_hits, images_misses, ((long double) images_hits / (long double) images_misses));
    printf("diffs  | hits: %lu, misses: %lu, ratio: %.2Lf\n", diffs_hits, diffs_misses, ((long double) diffs_hits / (long double) diffs_misses));

    cache_info_free(cache_info);

    destroy_db_info(db_info);

    return 0;
}

