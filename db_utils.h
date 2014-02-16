#ifndef DB_UTILS_H
#define DB_UTILS_H

#include <stdlib.h>
#include <libpq-fe.h>
#include "string.h"

typedef enum TilesState {
    TILES_ALREADY_EXISTS,
    NO_TILES_IN_DB,
    TILES_COUNT_MISMATCH
} TilesState;

typedef enum BufferState {
    BUFFER_EMPTY,
    BUFFER_FULL,
    BUFFER_OK
} BufferState;

typedef struct DbBuffer {
    char* buffer_str;
    size_t current_offset;
    size_t max_buffer_size;
} DbBuffer;

typedef struct DbInfo {
    PGconn* conn;
    DbBuffer* db_buffer;
} DbInfo;


DbInfo* create_db_info(PGconn* conn, size_t pg_sql_buffer_size);

void create_tables_if_not_exists(const DbInfo* const db_info);

void clear_all_data(const DbInfo *const db_info);
void clear_session_data(const DbInfo* const db_info);

void write_tiles_paths(const DbInfo* const db_info,
                                    const char* const* const paths,
                                    const unsigned int total_count,
                                    unsigned int* const ids_in_pg,
                                    void (*callback)(unsigned int));

void read_tiles_ids(const DbInfo* const db_info, unsigned int* const ids_in_pg);

TilesState check_tiles_in_db(const DbInfo *const db_info, unsigned int guess_count);

void add_tile_to_group(const DbInfo * const db_info, unsigned int leader_tile_id, unsigned int tile_id);
void add_tile_to_group_using_buffer(const DbInfo * const db_info, unsigned int leader_tile_id, unsigned int tile_id);

unsigned int load_next_zero_equal_id_leader(const DbInfo* const db_info);

void load_zero_equals_ids_for_tile(const DbInfo* const db_info,
                                   const unsigned int tile_id,
                                   unsigned int ** const ids_in_pg,
                                   unsigned int * const count);

void delete_zero_equal_pair_using_buffer(const DbInfo* const db_info,
                                         const unsigned int leader_tile_id,
                                         const unsigned int right_tile_id);

void delete_zero_equal_pair(const DbInfo* const db_info,
                            const unsigned int leader_tile_id,
                            const unsigned int right_tile_id);

void remix_zero_equals_ids(const DbInfo* const db_info, const unsigned int tile_id);

void add_tile_to_persistent_group(const DbInfo* const db_info,
                                const unsigned int leader_tile_id,
                                const unsigned int tile_id);

void add_tile_to_persistent_group_using_buffer(const DbInfo* const db_info,
                                               const unsigned int leader_tile_id,
                                               const unsigned int tile_id);

void destroy_db_info(DbInfo* const db_info);

void add_to_buffer(const DbInfo* const db_info, const char * const sql, const size_t offset);
BufferState get_buffer_state(const DbInfo* const db_info, const size_t chars_to_insert);


void insert_tile_color_using_buffer(const unsigned int tile_id,
                                    const unsigned int color,
                                    const unsigned int repeat_count,
                                    const DbInfo *const db_info);

void add_2_values_using_buffer(const DbInfo* const db_info,
                               const char* const sql_template_insert,
                               const char* const sql_template_values,
                               const unsigned int val0,
                               const unsigned int val1);

void flush_db_buffer(const DbInfo* const db_info);

void create_working_set(const DbInfo* const db_info);
void create_working_set_wo_equality_records(const DbInfo* const db_info);
void create_working_set_wo_persistent_ids(const DbInfo* const db_info);

void remove_tile_from_working_set(const DbInfo* const db_info, const unsigned int tile_id);
void remove_tile_and_equals_tiles_from_working_set(const DbInfo* const db_info, const unsigned int tile_id);
void remove_tile_and_equals_tiles_from_working_set_huge(const DbInfo* const db_info, const unsigned int tile_id);

unsigned int get_next_tile_from_working_set(const DbInfo* const db_info);
void read_working_set_ids(const DbInfo* const db_info,
                          unsigned int* const ids,
                          unsigned int* const count);


void materialize_count_equality(const DbInfo* const db_info);

void materialize_tile_color_count(const DbInfo* const db_info);
void materialize_tile_color_count_wo_persistent(const DbInfo* const db_info);

void read_related_tiles_ids(const DbInfo* const db_info, const unsigned int related_tile_id,
                            unsigned int *const ids,
                            unsigned int * const count,
                            const unsigned int max_diff_pixels);

void read_tile_color_count_tiles_ids(const DbInfo* const db_info, unsigned int *const ids, unsigned int* const count);

void read_persistent_groups(const DbInfo* const db_info,
                            unsigned int ** const leaders_ids,
                            unsigned int* const count);

unsigned char check_is_table_exist(const DbInfo* const db_info, const char* const table_name);

static inline void exec_no_result(const DbInfo* const db_info, const char* const sql) {
    PQclear(PQexec(db_info->conn, sql));
}

static inline void clear_working_set(const DbInfo* const db_info) {
    exec_no_result(db_info, "TRUNCATE working_set;");
}

static inline void clear_tile_color_count(const DbInfo* const db_info) {
    exec_no_result(db_info, "TRUNCATE tile_color_count_mv;");
}

static inline void drop_index_tile_color(const DbInfo* const db_info) {
    exec_no_result(db_info, "\
                    DROP INDEX tile_color_color_indx; \
                    DROP INDEX tile_color_color_repeat_indx; \
                    DROP INDEX tile_color_tile_id_indx;");
}

static inline void create_index_tile_color(const DbInfo* const db_info) {
    exec_no_result(db_info, "\
                CREATE INDEX tile_color_color_indx \
                    ON tile_color \
                    USING btree \
                    (color); \
                CREATE INDEX tile_color_color_repeat_indx \
                    ON tile_color \
                    USING btree \
                    (color, repeat_count); \
                CREATE INDEX tile_color_tile_id_indx \
                    ON tile_color \
                    USING btree \
                    (tile_id);");
}

void exec_template_one_param(const DbInfo* const db_info,
                             const char* const sql_template,
                             const unsigned int param);

static const char exists_template[] = "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name='%s')";


static const char clear_session_data_sql[] = "TRUNCATE tiles_groups, working_set RESTART IDENTITY CASCADE;";


static const char clear_data_sql[] = "\
    TRUNCATE working_set,\
        persistent_tiles_groups,\
        count_equality_mv,\
        tile_color,\
        tile_color_count_mv,\
        tiles_groups,\
        tiles\
    RESTART IDENTITY CASCADE;";


static const char sql_template_load_zero_equals_ids[] = "SELECT right_tile_id FROM count_equality_mv WHERE left_tile_id = %u;";

static const char sql_template_insert_tile_color[] = "INSERT INTO tile_color (tile_id, color, repeat_count) VALUES (%u,%u,%u)";
static const char sql_template_values_tile_color[] = ",(%u,%u,%u)";


static const char sql_template_insert_persistent_group_tile[] = "INSERT INTO persistent_tiles_groups(leader_tile_id, tile_id) VALUES(%u,%u);";
static const char sql_buffer_template_insert_persistent_group_tile[] = "INSERT INTO persistent_tiles_groups(leader_tile_id, tile_id) VALUES(%u,%u)";
static const char sql_buffer_template_values_persistent_group_tile[] = ",(%u,%u)";

static const char sql_template_delete_zero_equal_pair[] = "DELETE FROM count_equality_mv WHERE left_tile_id = %u AND right_tile_id = %u;";
static const char sql_buffer_template_delete_zero_equal_pair[] = "DELETE FROM count_equality_mv WHERE left_tile_id = %u AND right_tile_id IN (%u";
static const char sql_buffer_template_in_zero_equal_pair[] = ",%u";


static const char sql_template_add_tile_to_group[] = "INSERT INTO tiles_groups (leader_tile_id, tile_id) VALUES(%u,%u);";
static const char sql_buffer_template_insert_tile_to_group[] = "INSERT INTO tiles_groups (leader_tile_id, tile_id) VALUES(%u,%u)";
static const char sql_buffer_template_values_tile_to_group[] = ",(%u,%u)";

static const char sql_template_delete_tile_from_working_set[] = "DELETE FROM working_set WHERE tile_id = %u;";

static const char create_table_tiles[] = "\
    CREATE TABLE tiles (\
        id serial NOT NULL,\
        tile_path character varying(255) NOT NULL,\
        CONSTRAINT id_pk PRIMARY KEY (id)\
    );\
    CREATE UNIQUE INDEX tiles_tile_id_indx\
    ON tiles\
    USING btree\
    (id);";


static const char create_table_persistent_tiles_groups[] = "\
    CREATE TABLE persistent_tiles_groups\
    (\
        leader_tile_id integer NOT NULL,\
        tile_id integer NOT NULL,\
        CONSTRAINT leader_tile_id_fk FOREIGN KEY (leader_tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE,\
        CONSTRAINT tile_id_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );\
    CREATE INDEX persistent_tiles_groups_leader_tile_id_indx\
    ON persistent_tiles_groups\
    USING btree\
    (leader_tile_id );\
    \
    CREATE INDEX persistent_tiles_groups_tile_id_indx\
    ON persistent_tiles_groups\
    USING btree\
    (tile_id );";


static const char create_table_tiles_groups[] = "\
    CREATE TABLE tiles_groups(\
        leader_tile_id integer NOT NULL,\
        tile_id integer NOT NULL,\
        CONSTRAINT leader_tile_id_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE,\
        CONSTRAINT tile_id_fk FOREIGN KEY (tile_id)\
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
    \
    CREATE INDEX tile_color_color_indx \
        ON tile_color \
        USING btree \
        (color); \
    CREATE INDEX tile_color_color_repeat_indx\
        ON tile_color\
        USING btree\
        (color, repeat_count);\
    CREATE INDEX tile_color_tile_id_indx\
        ON tile_color\
        USING btree\
        (tile_id );";


static const char create_table_working_set[] = "\
    CREATE TABLE working_set ( \
        tile_id integer NOT NULL, \
        CONSTRAINT tile_id_fk FOREIGN KEY (tile_id) \
            REFERENCES tiles (id) MATCH SIMPLE \
            ON UPDATE CASCADE ON DELETE CASCADE \
    ); \
    \
    CREATE INDEX working_set_tile_id_indx \
        ON working_set \
        USING btree \
        (tile_id);";


static const char create_tile_color_records_count_view[] = "\
    CREATE OR REPLACE VIEW tile_color_records_count_view AS \
        SELECT tile_color.tile_id, count(*) AS count\
        FROM tile_color\
        GROUP BY tile_color.tile_id;";


static const char create_table_count_equality_mv[] = "\
    CREATE TABLE count_equality_mv (\
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
    CREATE INDEX count_equality_mv_left_tile_id_indx\
        ON count_equality_mv\
        USING btree\
        (left_tile_id );\
    \
    CREATE INDEX count_equality_mv_right_tile_id_indx\
        ON count_equality_mv\
        USING btree\
        (right_tile_id );";


static const char create_tile_color_count_mv_table[] = "\
    CREATE TABLE tile_color_count_mv (\
        tile_id integer,\
        count integer,\
        CONSTRAINT tile_id_fk FOREIGN KEY (tile_id)\
            REFERENCES tiles (id) MATCH SIMPLE\
            ON UPDATE CASCADE ON DELETE CASCADE\
    );\
    \
    CREATE INDEX tile_color_count_mv_tile_id_indx\
        ON tile_color_count_mv\
        USING btree\
        (tile_id );\
    \
    CREATE INDEX tile_color_count_mv_count_indx\
      ON tile_color_count_mv\
      USING btree\
      (count );";


static const char materializations_of_count_equality_mv_sql_template[] = "\
    INSERT INTO count_equality_mv ( \
        SELECT %u, tc1.tile_id \
        FROM tile_color tc0 \
        JOIN tile_color tc1 ON tc0.color = tc1.color \
            AND tc0.repeat_count = tc1.repeat_count \
            AND tc0.tile_id <> tc1.tile_id \
        WHERE tc0.tile_id = %u \
        GROUP BY tc1.tile_id \
        HAVING COUNT(tc0.color) = (SELECT count FROM tile_color_count_mv WHERE tile_id = tc1.tile_id));";


static const char remove_tile_and_equals_from_working_set_sql_template[] = "\
        DELETE FROM working_set \
        WHERE tile_id = %u \
            OR tile_id IN (SELECT t.tile_id \
                           FROM working_set t \
                           JOIN count_equality_mv ce ON ce.left_tile_id = t.tile_id \
                               OR ce.right_tile_id = t.tile_id);";

static const char remove_tile_and_equals_tiles_from_working_set_huge_sql_template[] = "\
    SELECT tile_id \
    INTO TEMP tmp_working_set \
    FROM \
        (SELECT tile_id \
        FROM working_set \
        EXCEPT \
        ( \
            (SELECT tile_id \
            FROM working_set ws \
            WHERE EXISTS (SELECT 1 FROM count_equality_mv ce WHERE ce.right_tile_id = ws.tile_id)) \
            \
            UNION ALL \
            \
            (SELECT %u) \
            \
            UNION ALL \
            \
            (SELECT DISTINCT left_tile_id FROM count_equality_mv) \
        ) \
    ) AS good_ids; \
    \
    TRUNCATE working_set; \
    INSERT INTO working_set (SELECT tile_id FROM tmp_working_set); \
    DROP TABLE tmp_working_set;";


static const char sql_template_remix_count_equality_mv[] = "\
    INSERT INTO count_equality_mv (\
        WITH right_ids AS \
            (SELECT right_tile_id\
            FROM count_equality_mv\
            WHERE left_tile_id = %u \
            ORDER BY right_tile_id)\
        \
        SELECT\
        (SELECT right_tile_id FROM right_ids LIMIT 1) AS left_tile_id,\
        right_tile_id\
        FROM right_ids OFFSET 1);\
 \
    DELETE FROM count_equality_mv WHERE left_tile_id = %u;";


// Postgresql WITH keyword save memory and reduce execution time
static const char materialization_of_tile_color_count_without_persistent[] = "\
    INSERT INTO tile_color_count_mv\
    (WITH available_tiles AS (\
    (\
    (SELECT id AS tile_id FROM tiles)\
        EXCEPT\
    (SELECT DISTINCT leader_tile_id FROM persistent_tiles_groups)\
    )\
        EXCEPT\
    (SELECT tile_id FROM persistent_tiles_groups)\
    )\
    SELECT tile_id, count FROM tile_color_records_count_view\
    WHERE tile_id IN (SELECT tile_id FROM available_tiles));";


static const char select_related_tiles_ids_sql_template[] = "\
    WITH tc0_colors AS \
        (SELECT tile_id, color, repeat_count \
        FROM tile_color \
        WHERE tile_id = %u) \
    SELECT tc1.tile_id \
    FROM tc0_colors tc0 \
    JOIN tile_color tc1 ON tc0.color = tc1.color \
    WHERE tc0.tile_id = %u \
        AND tc0.tile_id <> tc1.tile_id \
    GROUP BY tc0.tile_id, tc1.tile_id \
    HAVING SUM(tc0.repeat_count) >= %u AND SUM(tc1.repeat_count) >= %u;";

#endif // DB_UTILS_H
