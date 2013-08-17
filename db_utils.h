#ifndef DB_UTILS_H
#define DB_UTILS_H

#include <stdlib.h>
#include <libpq-fe.h>
#include "string.h"

#define TILES_ALREADY_EXISTS 1
#define NO_TILES_IN_DB 2
#define TILES_COUNT_MISMATCH 3

typedef struct DbBuffer {
    char* buffer_str;
    size_t current_offset;
    size_t max_buffer_size;
} DbBuffer;

typedef struct DbInfo {
    PGconn* conn;
    unsigned int last_virtual_group_id;
    DbBuffer* db_buffer;
} DbInfo;


DbInfo* init_db_info(PGconn* conn);

void create_tables_if_not_exists(const DbInfo* const db_info);

void clear_all_data(const DbInfo *const db_info);

void write_tiles_paths_to_pg(const DbInfo* const db_inf,
                             char **const paths,
                             unsigned int total_count,
                             unsigned int* const ids_in_pg,
                             void (*callback)(unsigned int));

void read_tiles_ids(const DbInfo* const db_info, unsigned int* const ids_in_pg);

unsigned char check_tiles_in_db(const DbInfo *const db_info, unsigned int guess_count);

unsigned int create_group(DbInfo *db_info, unsigned int leader_tile_id, unsigned int node_id);
unsigned int create_virtual_group(DbInfo* db_info);

void add_tile_to_group(const DbInfo* const db_info, unsigned int group_id, unsigned int tile_id);

void delete_group(DbInfo *db_info, unsigned int group_id);

void delete_db_info(DbInfo* db_info);

void flush_buffer_tiles_colors(const DbInfo* const db_info);

void insert_tile_color(const unsigned int tile_id,
                       const unsigned int color,
                       const unsigned int repeat_count,
                       const DbInfo *const db_info);

static inline void drop_index_tile_color(const DbInfo* const db_info) {
    PQclear(PQexec(db_info->conn, "DROP INDEX color_repeat_indx;"));
}

static inline void create_index_tile_color(const DbInfo* const db_info) {
    PQclear(PQexec(db_info->conn, "CREATE INDEX color_repeat_indx\
                   ON tile_color\
                   USING btree\
                   (color, repeat_count);"));
}

static inline void begin_transaction(const DbInfo* const db_info) {
    PQclear(PQexec(db_info->conn, "BEGIN;"));
}

static inline void commit_transaction(const DbInfo* const db_info) {
    PQclear(PQexec(db_info->conn, "COMMIT;"));
}


static const char sql_template_insert_tile_color[] = "INSERT INTO tile_color (tile_id, color, repeat_count) VALUES (%u,%u,%u)";
static const char sql_template_values_tile_color[] = ",(%u,%u,%u)";


static const char create_table_tiles[] = "CREATE TABLE tiles\
        (\
            id serial NOT NULL,\
            tile_path character varying(255) NOT NULL,\
            CONSTRAINT id_pk PRIMARY KEY (id)\
            )\
        WITH (\
            OIDS=FALSE\
        );\
ALTER TABLE tiles OWNER TO postgres;\
CREATE UNIQUE INDEX tile_id_indx\
ON tiles\
USING btree\
(id);";


static const char create_table_group[] = "CREATE TABLE tiles_groups\
        (\
            id serial NOT NULL, \
            leader_tile integer NOT NULL, \
            node_id integer NOT NULL, \
            CONSTRAINT tiles_groups_pk PRIMARY KEY (id), \
            CONSTRAINT leader_tile_fk FOREIGN KEY (leader_tile) REFERENCES tiles (id) ON UPDATE CASCADE ON DELETE CASCADE\
            ) \
        WITH (\
            OIDS = FALSE\
        )\
        ;\
ALTER TABLE tiles_groups OWNER TO postgres;\
\
CREATE INDEX left_tile_indx\
ON diff_graph\
USING btree\
(left_tile);\
\
CREATE INDEX right_tile_indx\
ON diff_graph\
USING btree\
(right_tile);";


static const char create_table_tile_group[] = "CREATE TABLE tile_group\
        (\
            tile_id integer NOT NULL,\
            group_id integer NOT NULL,\
            CONSTRAINT group_fk FOREIGN KEY (group_id)\
            REFERENCES tiles_groups (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE,\
            CONSTRAINT tile_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
            )\
        WITH (\
            OIDS=FALSE\
        );\
ALTER TABLE tile_group OWNER TO postgres;";


static const char create_table_tile_color[] = "CREATE TABLE tile_color\
        (\
            tile_id integer NOT NULL,\
            color bigint NOT NULL,\
            repeat_count integer NOT NULL,\
            CONSTRAINT tile_id_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
            )\
        WITH (\
            OIDS=FALSE\
        );\
ALTER TABLE tile_color OWNER TO postgres;\
\
CREATE INDEX color_repeat_indx\
ON tile_color\
USING btree\
(color, repeat_count);";

#endif // DB_UTILS_H
