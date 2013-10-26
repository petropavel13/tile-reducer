#ifndef DB_UTILS_H
#define DB_UTILS_H

#include <stdlib.h>
#include <libpq-fe.h>
#include "string.h"

#define TILES_ALREADY_EXISTS 1
#define NO_TILES_IN_DB 2
#define TILES_COUNT_MISMATCH 3

#define TABLE_DOESNT_EXIST 0
#define TABLE_ALREADY_EXIST 1

#define DOESNT_HAVE_COLOR 0
#define HAS_COLOR 1

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


DbInfo* init_db_info(PGconn* conn, size_t pg_sql_buffer_size);

void create_tables_if_not_exists(const DbInfo* const db_info);

void clear_all_data(const DbInfo *const db_info);
void clear_session_data(const DbInfo* const db_info);

void write_tiles_paths_to_pg(const DbInfo* const db_inf,
                             char **const paths,
                             unsigned int total_count,
                             unsigned int* const ids_in_pg,
                             void (*callback)(unsigned int));

void read_tiles_ids(const DbInfo* const db_info, unsigned int* const ids_in_pg);

unsigned char check_tiles_in_db(const DbInfo *const db_info, unsigned int guess_count);

unsigned int create_group(DbInfo *db_info, unsigned int leader_tile_id);
unsigned int create_virtual_group(DbInfo* db_info);

unsigned int create_persistent_group(const DbInfo* const db_info);
void add_tile_to_persistent_group(const DbInfo* const db_info,
                                 unsigned int tile_id,
                                 unsigned int persistent_group_id);

void create_new_persistent_group_from_parent(const DbInfo* const db_info,
                                             unsigned int parent_persistent_group,
                                             unsigned int color,
                                             unsigned char has_marker);

void read_colors(const DbInfo* const db_info, unsigned int** colors, unsigned int* count);

void add_tile_to_group(const DbInfo* const db_info,
                       unsigned int group_id,
                       unsigned int tile_id);

void delete_group(DbInfo *db_info, unsigned int group_id);

void delete_db_info(DbInfo* db_info);

void flush_buffer_tiles_colors(const DbInfo* const db_info);

void insert_tile_color(const unsigned int tile_id,
                       const unsigned int color,
                       const unsigned int repeat_count,
                       const DbInfo *const db_info);

static const char exists_template[] = "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name='%s')";

unsigned char check_is_table_exist(const DbInfo* const db_info, const char* const table_name);

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


static const char create_table_tiles[] = "\
    CREATE TABLE tiles (\
        id serial NOT NULL,\
        tile_path character varying(255) NOT NULL,\
        CONSTRAINT id_pk PRIMARY KEY (id)\
    );\
    CREATE UNIQUE INDEX tile_id_indx\
    ON tiles\
    USING btree\
    (id);";


static const char create_table_persistent_groups[] = "\
    CREATE TABLE persistent_groups (\
        id serial NOT NULL,\
        CONSTRAINT persistent_group_pk PRIMARY KEY (id )\
    );";


static const char create_table_persistent_group_tile[] = "\
    CREATE TABLE persistent_group_tile (\
        group_id integer NOT NULL,\
        tile_id integer NOT NULL,\
        CONSTRAINT group_fk FOREIGN KEY (group_id)\
            REFERENCES persistent_groups (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE,\
        CONSTRAINT tile_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );";


static const char create_table_tile_group[] = "\
    CREATE TABLE tile_group (\
        tile_id integer NOT NULL,\
        group_id integer NOT NULL,\
        CONSTRAINT group_fk FOREIGN KEY (group_id)\
            REFERENCES tiles_groups (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE,\
        CONSTRAINT tile_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );";


static const char create_table_tiles_groups[] = "\
    CREATE TABLE tiles_groups (\
        id serial NOT NULL,\
        leader_tile integer NOT NULL,\
        CONSTRAINT tiles_groups_pk PRIMARY KEY (id ),\
        CONSTRAINT leader_tile_fk FOREIGN KEY (leader_tile)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );";


static const char create_table_tile_color[] = "\
    CREATE TABLE tile_color (\
        tile_id integer NOT NULL,\
        color bigint NOT NULL,\
        repeat_count integer NOT NULL,\
        CONSTRAINT tile_id_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );\
    CREATE INDEX color_repeat_indx\
    ON tile_color\
    USING btree\
    (color, repeat_count);";


static const char create_table_working_set[] = "\
    CREATE TABLE working_set (\
        tile_id integer NOT NULL,\
        CONSTRAINT tile_id_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );";

#endif // DB_UTILS_H
