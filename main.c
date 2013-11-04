#include <stdio.h>

#include "tile_utils.h"
#include "cache_utils.h"
#include "db_utils.h"
#include "cluster_utils.h"
#include "color_index_utils.h"

#include <libpq-fe.h>

#define SLEEP_MICSECS_ON_CUDA_ERROR 8000000

#define DEFAULT_MB_IMAGE_CACHE_SIZE 512
#define DEFAULT_MB_DIFF_CACHE_SIZE 128
#define DEFAULT_MB_PG_SQL_BUFFER_SIZE 8

//#define DEBUG 1

void print_progress_paths_read(const unsigned char percent_done) {
    printf("\r                            ");
    printf("\rReading tiles paths...%d%%", percent_done);
    fflush(stdout);
}

void print_progress_tiles_db_write(unsigned int current) {
    printf("\r                            ");
    printf("\rWriting tiles paths to DB...%d", current);
    fflush(stdout);
}

void print_progress_tiles_colors_db_write(const unsigned char percent_done) {
    printf("\r                            ");
    printf("\rWriting tiles colors to db...%d%%", percent_done);
    fflush(stdout);
}

int main(int argc, char* argv [])
{
    if(argc < 2) {
        printf("-- too few parameters! --\n");
        printf("template: ./comparer /path/to/tiles/folder/ MAX_DIFF_PIXELS [MAX_MB_CACHE]\n");
        printf("example: ./comparer /tiles/opt_easy/ 16 4096\n");
        printf("example: ./comparer /tiles/opt_easy/ 64\n");

        return 1;
    }

    const char* path = argv[1];
    const unsigned short int max_diff_pixels = atoi(argv[2]);
    const size_t max_cache_size_bytes = (argc > 3) ? ((size_t) atoi(argv[3])) * 1024 * 1024 : ((size_t) DEFAULT_MB_IMAGE_CACHE_SIZE) * 1024 * 1024;

    printf("Tiles folder: \"%s\";\nmax_diff_pixels: %d;\ncache_size: %d MB;\n\n", path, max_diff_pixels, (unsigned int) (max_cache_size_bytes / 1024 / 1024));

    printf("Computing tiles count...");
    fflush(stdout);

    const unsigned int total = get_total_files_count(path);

    printf("\rTotal tiles count: %d         \n", total);
    fflush(stdout);

//    PGconn* conn = PQconnectdb("dbname=tiles_db hostaddr=192.168.0.39 user=postgres port=5432 password=123");
    PGconn* conn = PQconnectdb("dbname=tiles_db hostaddr=172.18.36.131 user=postgres port=5432 password=123");
//    PGconn* conn = PQconnectdb("dbname=tiles_db host=/var/run/postgresql user=postgres password=123");

    printf("Connecting to db...");
    fflush(stdout);

    if (PQstatus(conn) == CONNECTION_BAD) {
        printf("%s\n", PQerrorMessage(conn));

        return 1;
    }

    DbInfo* const db_info = init_db_info(conn, 1024 * 1024 * DEFAULT_MB_PG_SQL_BUFFER_SIZE);

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
        write_tiles_paths(db_info, tiles_paths, total, pg_ids, &print_progress_tiles_db_write);

        printf("\r                                                ");
        printf("\rWriting tiles paths to DB...done\n");
        fflush(stdout);
    } else if(res == TILES_COUNT_MISMATCH) {
        printf("Tiles count mismatch. Clean up db.\n");
        fflush(stdout);

        clear_all_data(db_info);
        write_tiles_paths(db_info, tiles_paths, total, pg_ids, &print_progress_tiles_db_write);

        printf("\r                                                ");
        printf("\rWriting tiles paths to DB...done\n");
        fflush(stdout);
    }

    CacheInfo* const cache_info = init_cache(max_cache_size_bytes, DEFAULT_MB_DIFF_CACHE_SIZE * 1024 * 1024, TILE_SIZE_BYTES);

    TreeInfo* tree_info = (TreeInfo*)malloc(sizeof(TreeInfo));
    tree_info->data_destructor = NULL;
    GenericNode* root_node = NULL;

    GroupElement* tiles_sequence = malloc(sizeof(GroupElement));
    tiles_sequence->first = tiles_sequence;

    GroupElement* prev = NULL;

    Tile* temp_tile = NULL;

    unsigned int current = 0;
    unsigned char last_percent = 0;

    for (unsigned int i = 0; i < total; ++i)
    {
        temp_tile = malloc(sizeof(Tile));
        temp_tile->tile_id = pg_ids[i];
        temp_tile->tile_file = read_tile(tiles_paths[i]);

        if(root_node != NULL) {
            root_node = insert(root_node, temp_tile->tile_id, temp_tile);
        } else {
            root_node = create_node(temp_tile->tile_id, temp_tile);
        }

        tiles_sequence->node = temp_tile;
        tiles_sequence->next = malloc(sizeof(GroupElement));
        tiles_sequence->next->first = tiles_sequence->first;

        prev = tiles_sequence;

        tiles_sequence = tiles_sequence->next;

        const unsigned char current_percent = (++current / (total / 100));

        if(last_percent != current_percent) {
            printf("\r                                                ");
            printf("\rLoading tiles into RAM...%d%%", current_percent);
            fflush(stdout);
            last_percent = current_percent;
        }
    }

    prev->next = NULL;
    free(tiles_sequence);
    tiles_sequence = prev;


    printf("\r                                                ");
    printf("\rLoading tiles into RAM...done\n");
    fflush(stdout);

    for (unsigned int i = 0; i < total; ++i)
    {
        free(tiles_paths[i]);
    }

    free(tiles_paths);

    if(res != TILES_ALREADY_EXISTS) {
        TilesTree* const tiles_tree = init_tiles_tree();

        GroupElement* temp_group_elem = tiles_sequence->first;

        for (int i = 0; temp_group_elem != NULL; ++i) {
            printf("\r                                                ");
            printf("\rIndexing tiles colors...%d", i);
            fflush(stdout);

            index_tile(temp_group_elem->node, tiles_tree);

            temp_group_elem = temp_group_elem->next;
        }

        printf("\r                                                ");
        printf("\rIndexing tiles colors...done\n");
        fflush(stdout);

        flush_tiles_colors_tree(tiles_tree, db_info, &print_progress_tiles_colors_db_write);

        printf("\r                                                ");
        printf("\rWriting tiles colors to db...done\n");
        fflush(stdout);

        destroy_tile_color_tree(tiles_tree);

        printf("\r                                                ");
        printf("\rMaterializing count equality view...");
        fflush(stdout);

        materialize_count_equality_view(db_info);

        printf("\r                                                ");
        printf("\rMaterializing count equality view...done\n");
        fflush(stdout);
    }

    make_persistent_groups(db_info, root_node, total, cache_info);
//    clusterize(tiles_sequence, total, max_diff_pixels, total / 2, cache_info, db_info);

    destroy_tree(root_node, tree_info);
    free(tree_info);

    const int images_hits = cache_info->image_hit_count;
    const int images_misses = cache_info->image_miss_count;

    const int diffs_hits = cache_info->edges_hit_count;
    const int diffs_misses = cache_info->edges_miss_count;

    printf("images | hits: %d, misses: %d, ratio: %f\n", images_hits, images_misses, ((float) images_hits / (float) images_misses));
    printf("diffs  | hits: %d, misses: %d, ratio: %f\n", diffs_hits, diffs_misses, ((float) diffs_hits / (float) diffs_misses));

    delete_cache(cache_info);

    delete_db_info(db_info);

    PQfinish(conn);

    return 0;
}

