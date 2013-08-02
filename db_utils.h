#ifndef DB_UTILS_H
#define DB_UTILS_H

#include <stdlib.h>
#include <libpq-fe.h>

#define TILES_ALREADY_EXISTS 1
#define NO_TILES_IN_DB 2
#define TILES_COUNT_MISMATCH 3

void create_tables_if_not_exists(PGconn* conn);

void clear_all_data(PGconn* conn);

void write_tiles_paths_to_pg(PGconn *conn,
                             char **const paths,
                             unsigned int total_count,
                             unsigned int* const ids_in_pg,
                             void (*callback)(unsigned int, unsigned int, const char*));

void read_tiles_ids(PGconn *conn, unsigned int* const ids_in_pg);

unsigned int check_tiles_in_db(PGconn *conn, unsigned int guess_count);

unsigned int create_group(PGconn *conn, unsigned int leader_tile_id, unsigned int node_id);

void add_tile_to_group(PGconn *conn, unsigned int group_id, unsigned int tile_id);

void delete_group(PGconn *conn, unsigned int group_id);

#endif // DB_UTILS_H
