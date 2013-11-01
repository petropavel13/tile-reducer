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

#define STRING_TEMPLATE_SIZE 2
#define MAX_UINT32_STR_LEN 10


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


void load_zero_equals_ids(const DbInfo* const db_info,
                          unsigned int tile_id,
                          unsigned int* ids_in_pg, unsigned int *count);
unsigned int create_persistent_group(const DbInfo* const db_info, unsigned int leader_tile_id);
void add_tile_to_persistent_group(const DbInfo* const db_info,
                                 unsigned int tile_id,
                                 unsigned int persistent_group_id);


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

void create_working_set(const DbInfo* const db_info);
void create_working_set_wo_persistent_records(const DbInfo* const db_info);
void clear_working_set(const DbInfo* const db_info);
unsigned int get_next_tile_from_working_set(const DbInfo* const db_info);
void remove_tile_from_working_set(const DbInfo* const db_info, unsigned int tile_id);

void materialize_count_equality_view(const DbInfo* const db_info);

void materialize_tile_color_count(const DbInfo* const db_info);

unsigned char check_is_table_exist(const DbInfo* const db_info, const char* const table_name);

static inline void exec_no_result(const DbInfo* const db_info, const char * sql) {
    PQclear(PQexec(db_info->conn, sql));
}

static inline void drop_index_tile_color(const DbInfo* const db_info) {
    exec_no_result(db_info, "DROP INDEX tile_color_color_repeat_indx;");
}

static inline void create_index_tile_color(const DbInfo* const db_info) {
    exec_no_result(db_info, "CREATE INDEX tile_color_color_repeat_indx\
                   ON tile_color\
                   USING btree\
                   (color, repeat_count);");
}

static inline void begin_transaction(const DbInfo* const db_info) {
    exec_no_result(db_info, "BEGIN;");
}

static inline void commit_transaction(const DbInfo* const db_info) {
    exec_no_result(db_info, "COMMIT;");
}

static const char exists_template[] = "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name='%s')";

static const char clear_session_data_sql[] = "TRUNCATE tile_group, tiles_groups, working_set;";
static const char restart_session_sequences[] = "ALTER SEQUENCE tiles_groups_id_seq RESTART WITH 1;";

static const char clear_data_sql[] = "\
    TRUNCATE working_set,\
            persistent_groups,\
            persistent_group_tile,\
            materialized_count_equality_view,\
            tile_color,\
            tile_group,\
            tiles_groups,\
            tiles;";

static const char restart_sequences[] = "\
    ALTER SEQUENCE tiles_id_seq RESTART WITH 1;\
    ALTER SEQUENCE persistent_groups_id_seq RESTART WITH 1;\
    ALTER SEQUENCE tiles_groups_id_seq RESTART WITH 1;";

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
    CREATE TABLE persistent_groups(\
        id serial NOT NULL,\
        leader_tile_id integer,\
        CONSTRAINT persistent_group_pk PRIMARY KEY (id ),\
        CONSTRAINT leader_tile_id_fk FOREIGN KEY (leader_tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    )";


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
        leader_tile_id integer NOT NULL,\
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
    CREATE INDEX tile_color_color_repeat_indx\
        ON tile_color\
        USING btree\
        (color, repeat_count);\
    CREATE INDEX tile_color_tile_id_indx\
        ON tile_color\
        USING btree\
        (tile_id );";


static const char create_table_working_set[] = "\
    CREATE TABLE working_set (\
        tile_id integer NOT NULL,\
        CONSTRAINT tile_id_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );";


static const char create_tile_color_records_count_view[] = "\
    CREATE OR REPLACE VIEW tile_color_records_count_view AS \
        SELECT tile_color.tile_id, count(*) AS count\
        FROM tile_color\
        GROUP BY tile_color.tile_id;";


// Postgresql WITH keyword save memory and reduce execution time
static const char create_count_equality_view[] = "\
    CREATE OR REPLACE VIEW count_equality_view AS \
    WITH  tcc AS (SELECT tile_id, count FROM tile_color_count)\
    SELECT t0.tile_id AS left_tile_id, t1.tile_id AS right_tile_id\
    FROM tcc t0\
    JOIN tcc t1 ON t0.count = t1.count\
    WHERE t0.tile_id <> t1.tile_id;";


static const char create_table_materialized_count_equality_view[] = "\
    CREATE TABLE materialized_count_equality_view (\
        left_tile_id integer NOT NULL,\
        right_tile_id integer NOT NULL,\
        CONSTRAINT left_tile_id_fk FOREIGN KEY (left_tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE,\
        CONSTRAINT right_tile_id FOREIGN KEY (right_tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );\
    \
    CREATE INDEX mjrev_left_tile_id_indx\
        ON materialized_count_equality_view\
        USING btree\
        (left_tile_id );\
    \
    CREATE INDEX mjrev_right_tile_id_indx\
        ON materialized_count_equality_view\
        USING btree\
        (right_tile_id );";

static const char create_tile_color_count_table[] = "\
    CREATE TABLE tile_color_count (\
        tile_id integer,\
        count integer,\
        CONSTRAINT tile_id_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    )\
    \
    CREATE INDEX tile_color_count_count_indx\
        ON tile_color_count\
        USING btree\
        (tile_id );\
    \
    CREATE INDEX tile_color_count_tile_id_indx\
        ON tile_color_count\
        USING btree\
        (tile_id );";

// Postgresql SELECT INTO keyword save memory and reduce execution time
static const char materialization_sql_template[] = "\
    DROP TABLE IF EXISTS matched_tiles;\
    \
    SELECT left_tile_id, right_tile_id\
    INTO TEMP matched_tiles\
    FROM count_equality_view\
    WHERE left_tile_id = %d;\
    \
    INSERT INTO materialized_count_equality_view (SELECT left_tile_id, right_tile_id FROM matched_tiles);\
    \
    DELETE FROM working_set WHERE tile_id IN (SELECT right_tile_id FROM matched_tiles);\
    DELETE FROM working_set WHERE tile_id = %d;\
    DELETE FROM tile_color_count WHERE tile_id IN (SELECT right_tile_id FROM matched_tiles);\
    DELETE FROM tile_color_count WHERE tile_id = %d;";

#endif // DB_UTILS_H
