#include <stdio.h>

#include "tile_utils.h"
#include "cache_utils.h"
#include "db_utils.h"
#include "cluster_utils.h"
#include "color_index_utils.h"
#include "apprunparams.h"

#include <libpq-fe.h>
#include <pthread.h>

#define DEFAULT_MB_IMAGE_CACHE_SIZE 512
#define DEFAULT_MB_DIFF_CACHE_SIZE 128
#define DEFAULT_MB_PG_SQL_BUFFER_SIZE 8
#define DEFAULT_NUM_THREADS 4

#define STRINGS_EQUAL 0

#define LEFT 0
#define RIGHT 1
//#define DEBUG 1

char* get_arg(const int argc, char** argv, const char* const key) {
    int i = 0;

    for(; i < argc; ++i) {
        if (strstr(argv[i], key) != NULL) {
            if (strchr(argv[i], '=') != NULL) {
                return &(argv[i][strlen(key) + 1]);
            } else {
                return "true";
            }
        }
    }

    return "false";
}

void print_progress_paths_read(const unsigned char percent_done) {
    printf("\r                            ");
    printf("\rReading tiles paths...%d%%", percent_done);
    fflush(stdout);
}

void print_progress_tiles_db_write(const unsigned int current) {
    printf("\r                            ");
    printf("\rWriting tiles paths to DB...%d", current);
    fflush(stdout);
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


void run_index_threads(GenericNode* const tiles_tree, const unsigned char num_threads) {
    pthread_t threads[num_threads];
    GenericNode * heads[num_threads];
    DbInfo * connections[num_threads];
    TCTParams th_params[num_threads];

    for (unsigned char i = 0; i < num_threads; ++i) {
        heads[i] = get_head(tiles_tree, num_threads >> 2, i);
        connections[i] = create_db_info(PQconnectdb("dbname=tiles_db hostaddr=192.168.0.108 user=postgres port=5432 password=123"), 1024 * 1024 * 1);
//        th_params[i] = make_tct_params(heads[i], NULL, NULL, connections[i]);
        th_params[i] = make_tct_params(heads[i], &print_progress_index_colors, &print_progress_tiles_colors_db_write, connections[i]);
        pthread_create(&threads[i], NULL, &index_tree_and_flush_result, &th_params[i]);
    }


    GenericNode* const tails = get_tails(create_node(tiles_tree->key, tiles_tree->data), tiles_tree, num_threads >> 2);

    DbInfo* const db_info = create_db_info(PQconnectdb("dbname=tiles_db hostaddr=192.168.0.39 user=postgres port=5432 password=123"), 1024 * 1024 * 1);
    TCTParams params = make_tct_params(tails, NULL, NULL, db_info);

    index_tree_and_flush_result(&params);

    destroy_tree(tails, NULL);
    destroy_db_info(db_info);

    for (unsigned char i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
        destroy_db_info(connections[i]);
    }
}

void print_help() {
    printf("template: ./comparer --path=/path/to/tiles/folder/ --max_diff_pixels=0...65535 [--max_mb_cache=] [--max_num_theads=2^x -> (2,4,8,16,...)]\n");
    printf("example: ./comparer --path=/tiles/opt_easy/ --max_diff_pixels=16 --max_mb_cache=4096 --max_threads=8\n");
    printf("example: ./comparer --path=/tiles/opt_easy/ --max_diff_pixels=64\n");
}


int main(int argc, char* argv [])
{
    const char* const path = get_arg(argc, argv, "--path");
    const char* const max_diff_pixels_param = get_arg(argc, argv, "--max_diff_pixels");

    if (strcmp(path, "false") == STRINGS_EQUAL || strcmp(max_diff_pixels_param, "false") == STRINGS_EQUAL) {
        print_help();
        return 1;
    }

    const unsigned short int max_diff_pixels = atoi(max_diff_pixels_param);
    const char* const max_cache_size_param = get_arg(argc, argv, "--max_mb_cache");

    const size_t max_cache_size_bytes = strcmp(max_cache_size_param, "false") != STRINGS_EQUAL ?
                ((size_t) atoi(max_cache_size_param)) * 1024 * 1024 :
                ((size_t) DEFAULT_MB_IMAGE_CACHE_SIZE) * 1024 * 1024;


    const char* const max_threads_param = get_arg(argc, argv, "--max_threads");

    const unsigned char max_num_threads = strcmp(max_threads_param, "false") != STRINGS_EQUAL ? atoi(max_threads_param) : DEFAULT_NUM_THREADS;

    if ((max_num_threads & (~max_num_threads + 1)) != max_num_threads) {
        printf("max_threads must be power of 2!\n\n");
        print_help();
        return 1;
    }

    const AppRunParams arp = make_app_run_params(max_diff_pixels, max_num_threads);

    printf("Tiles folder: \"%s\"\nMax diff. pixels: %u\nCache size: %u MB\nMax threads: %u\n\n",
                          path,
                          max_diff_pixels,
                          (unsigned int) (max_cache_size_bytes / 1024 / 1024),
                          (unsigned int) max_num_threads
        );

    printf("Computing tiles count...");
    fflush(stdout);

    const unsigned int total = get_total_files_count(path);

    printf("\rTotal tiles count: %d         \n", total);
    fflush(stdout);

    PGconn* const conn = PQconnectdb("dbname=tiles_db hostaddr=192.168.0.108 user=postgres port=5432 password=123");
//    PGconn* conn = PQconnectdb("dbname=tiles_db hostaddr=172.18.36.131 user=postgres port=5432 password=takeit4");
//    PGconn* conn = PQconnectdb("dbname=tiles_db host=/var/run/postgresql user=postgres password=123");

    printf("Connecting to db...");
    fflush(stdout);

    if (PQstatus(conn) == CONNECTION_BAD) {
        printf("%s\n", PQerrorMessage(conn));

        return 1;
    }

    DbInfo* const db_info = create_db_info(conn, 1024 * 1024 * DEFAULT_MB_PG_SQL_BUFFER_SIZE);

    printf("done\n");
    fflush(stdout);

    unsigned int zero = 0;
    unsigned char percent = 0;

    char** const tiles_paths = malloc(sizeof(char*) * total);

    read_tiles_paths(path, tiles_paths, &total, &zero, &percent, &print_progress_paths_read);

    printf("\r                                                ");
    printf("\rReading tiles paths...done\n");
    fflush(stdout);

    create_tables_if_not_exists(db_info);

//    clear_all_data(db_info);

    const unsigned int res = check_tiles_in_db(db_info, total);

    unsigned int* pg_ids = malloc(sizeof(unsigned int) * total);

    if(res == TILES_ALREADY_EXISTS) {
        printf("Tiles already in db. Reading ids...\n");

        clear_session_data(db_info);

        read_tiles_ids(db_info, pg_ids);
    } else if(res == NO_TILES_IN_DB) {
        clear_all_data(db_info); // reset sequences
        write_tiles_paths(db_info, (const char * const * const)tiles_paths, total, pg_ids, &print_progress_tiles_db_write);

        printf("\r                                                ");
        printf("\rWriting tiles paths to DB...done\n");
        fflush(stdout);
    } else if(res == TILES_COUNT_MISMATCH) {
        printf("Tiles count mismatch. Clean up db.\n");
        fflush(stdout);

        clear_all_data(db_info);
        write_tiles_paths(db_info, (const char * const * const)tiles_paths, total, pg_ids, &print_progress_tiles_db_write);

        printf("\r                                                ");
        printf("\rWriting tiles paths to DB...done\n");
        fflush(stdout);
    }

    CacheInfo* const cache_info = init_cache(max_cache_size_bytes, DEFAULT_MB_DIFF_CACHE_SIZE * 1024 * 1024, TILE_SIZE_BYTES);

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

    for (unsigned int i = 0; i < total; ++i)
    {
        free(tiles_paths[i]);
    }

    free(tiles_paths);

    if(res != TILES_ALREADY_EXISTS) {
        drop_index_tile_color(db_info);

        if (max_num_threads < 2) {
            TCTParams params = make_tct_params(all_tiles, &print_progress_index_colors, &print_progress_tiles_colors_db_write, db_info);
            index_tree_and_flush_result(&params);
        } else {
//            printf("Indexing and writing tiles color %u threads...", max_num_threads);
            fflush(stdout);
            run_index_threads(all_tiles, max_num_threads);
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

    if (max_diff_pixels > 0) {
        clusterize(all_tiles, total, arp, db_info, cache_info);
    }

    destroy_tree(all_tiles, &tile_destructor);

    const unsigned long images_hits = cache_info->image_hit_count;
    const unsigned long images_misses = cache_info->image_miss_count;

    const unsigned long diffs_hits = cache_info->edges_hit_count;
    const unsigned long diffs_misses = cache_info->edges_miss_count;

    printf("images | hits: %lu, misses: %lu, ratio: %.2Lf\n", images_hits, images_misses, ((long double) images_hits / (long double) images_misses));
    printf("diffs  | hits: %lu, misses: %lu, ratio: %.2Lf\n", diffs_hits, diffs_misses, ((long double) diffs_hits / (long double) diffs_misses));

    destroy_cache(cache_info);

    destroy_db_info(db_info);

    return 0;
}

