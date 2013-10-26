#include "db_utils.h"

DbInfo* init_db_info(PGconn* conn, size_t pg_sql_buffer_size) {
    DbInfo* db_info = malloc(sizeof(DbInfo));
    db_info->conn = conn;
    db_info->last_virtual_group_id = 0;
    db_info->db_buffer = malloc(sizeof(DbBuffer));
    db_info->db_buffer->current_offset = 0;
    db_info->db_buffer->max_buffer_size = pg_sql_buffer_size;
    db_info->db_buffer->buffer_str = malloc(sizeof(char) * db_info->db_buffer->max_buffer_size);

    return db_info;
}

unsigned char check_is_table_exist(const DbInfo* const db_info, const char* const table_name) {
    char sql[strlen(exists_template) + strlen(table_name) - 2];

    sprintf(sql, exists_template, table_name);

    PGresult *res;

    res = PQexec(db_info->conn, sql);

    unsigned char exists = PQgetvalue(res, 0, 0)[0] == 't';

    PQclear(res);

    return exists;
}

void create_tables_if_not_exists(const DbInfo* const db_info) {
    if(check_is_table_exist(db_info, "tiles") == TABLE_DOESNT_EXIST) {
        PQclear(PQexec(db_info->conn, create_table_tiles));
    }

    if(check_is_table_exist(db_info,"tiles_groups") == TABLE_DOESNT_EXIST) {
        PQclear(PQexec(db_info->conn, create_table_tiles_groups));
    }

    if(check_is_table_exist(db_info,"tile_group") == TABLE_DOESNT_EXIST) {
        PQclear(PQexec(db_info->conn, create_table_tile_group));
    }

    if(check_is_table_exist(db_info,"tile_color") == TABLE_DOESNT_EXIST) {
        PQclear(PQexec(db_info->conn, create_table_tile_color));
    }

    if(check_is_table_exist(db_info,"persistent_groups") == TABLE_DOESNT_EXIST) {
        PQclear(PQexec(db_info->conn, create_table_persistent_groups));
    }

    if(check_is_table_exist(db_info,"persistent_group_tile") == TABLE_DOESNT_EXIST) {
        PQclear(PQexec(db_info->conn, create_table_persistent_group_tile));
    }

    if(check_is_table_exist(db_info,"working_set") == TABLE_DOESNT_EXIST) {
        PQclear(PQexec(db_info->conn, create_table_working_set));
    }
}

void clear_all_data(const DbInfo* const db_info) {
    const char clear_data_sql[] = "TRUNCATE working_set, persistent_groups, persistent_group_tile, tile_color, tile_group, tiles_groups, tiles;";
    const char restart_sequences[] = "ALTER SEQUENCE tiles_id_seq RESTART WITH 1;\
                                        ALTER SEQUENCE persistent_groups_id_seq RESTART WITH 1;\
                                        ALTER SEQUENCE tiles_groups_id_seq RESTART WITH 1;";

    PQclear(PQexec(db_info->conn, clear_data_sql));
    PQclear(PQexec(db_info->conn, restart_sequences));
}

void clear_session_data(const DbInfo* const db_info) {
    const char clear_session_data_sql[] = "TRUNCATE tile_group, tiles_groups, working_set;";
    const char restart_session_sequences[] = "ALTER SEQUENCE tiles_groups_id_seq RESTART WITH 1;";

    PQclear(PQexec(db_info->conn, clear_session_data_sql));
    PQclear(PQexec(db_info->conn, restart_session_sequences));
}

void write_tiles_paths_to_pg(const DbInfo* const db_info,
                             char **const paths,
                             unsigned int total_count,
                             unsigned int *const ids_in_pg,
                             void (*callback) (unsigned int)) {
    begin_transaction(db_info);

    PGresult *res;

    const char sql_template_insert[] = "INSERT INTO tiles (tile_path) VALUES('%s')";
    const char sql_template_values[] = ",('%s')";
    const char sql_returning[] = " RETURNING id;";

    const size_t insert_template_len = strlen(sql_template_insert);
    const size_t values_template_len = strlen(sql_template_values);
    const size_t returning_len = strlen(sql_returning);

    const unsigned char max_path_diff_size = 16; // suppose path length varies by 16
    const size_t max_insert_len = strlen(sql_template_insert) + strlen(paths[0]) + max_path_diff_size;
    const size_t max_values_len = strlen(sql_template_values) + strlen(paths[0]) + max_path_diff_size;

    char sql_insert[max_insert_len];
    char sql_values[max_values_len];


    sprintf(sql_insert, sql_template_insert, paths[0]);

    size_t insert_len = insert_template_len + strlen(paths[0]) - 2;
    sql_insert[insert_len] = '\0';

    strncpy(db_info->db_buffer->buffer_str, sql_insert, insert_len);

    db_info->db_buffer->current_offset = insert_len;

    size_t values_len;

    unsigned int last_flush_index = 0;

    for (unsigned int i = 1; i < total_count; ++i)
    {
        if(db_info->db_buffer->current_offset + returning_len + 1 >= db_info->db_buffer->max_buffer_size) {
            if(callback != NULL) {
                callback(i);
            }

            strncpy(db_info->db_buffer->buffer_str + db_info->db_buffer->current_offset, sql_returning, returning_len);

            db_info->db_buffer->current_offset += returning_len;

            db_info->db_buffer->buffer_str[db_info->db_buffer->current_offset] = '\0';

            res = PQexec(db_info->conn, db_info->db_buffer->buffer_str);

            unsigned int row_index = 0;

            for (unsigned int j = last_flush_index; j < i; ++j) {
                ids_in_pg[j] = atoi(PQgetvalue(res, row_index++, 0));
            }

            PQclear(res);

            last_flush_index = i;

            insert_len = insert_template_len + strlen(paths[i]) - 2;

            sprintf(sql_insert, sql_template_insert, paths[i]);
            strncpy(db_info->db_buffer->buffer_str, sql_insert, insert_len);

            db_info->db_buffer->current_offset = insert_len;
        } else {
            sprintf(sql_values, sql_template_values, paths[i]);

            values_len = values_template_len + strlen(paths[i]) - 2;

            strncpy(db_info->db_buffer->buffer_str + db_info->db_buffer->current_offset, sql_values, values_len);

            db_info->db_buffer->current_offset += values_len;
        }
    }

    if(db_info->db_buffer->current_offset > 0) {
        strncpy(db_info->db_buffer->buffer_str + db_info->db_buffer->current_offset, sql_returning, returning_len);

        db_info->db_buffer->current_offset += returning_len;

        db_info->db_buffer->buffer_str[db_info->db_buffer->current_offset] = '\0';

        res = PQexec(db_info->conn, db_info->db_buffer->buffer_str);

        unsigned int row_index = 0;

        for (unsigned int j = last_flush_index; j < total_count; ++j) {
            ids_in_pg[j] = atoi(PQgetvalue(res, row_index++, 0));
        }

        db_info->db_buffer->current_offset = 0;

        PQclear(res);
    }

    commit_transaction(db_info);
}

void read_tiles_ids(const DbInfo *const db_info, unsigned int* const ids_in_pg) {
    const char select_ids_sql[] = "SELECT id FROM tiles;";

    PGresult *res;

    res = PQexec(db_info->conn, select_ids_sql);

    const unsigned int count_objects = PQntuples(res);

    for (unsigned int i = 0; i < count_objects; ++i) {
        ids_in_pg[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

unsigned char check_tiles_in_db(const DbInfo* const db_info, unsigned int guess_count) {
    const char count_sql[] = "SELECT COUNT(*) FROM tiles;";

    PGresult *res;

    res = PQexec(db_info->conn, count_sql);

    const unsigned int count_objects = atoi(PQgetvalue(res, 0, 0));

    PQclear(res);

    if(count_objects == 0) {
        return NO_TILES_IN_DB;
    } else if(count_objects == guess_count) {
        return TILES_ALREADY_EXISTS;
    } else {
        return TILES_COUNT_MISMATCH;
    }
}


unsigned int create_group(DbInfo *db_info, unsigned int leader_tile_id) {
    PGresult *res;

    const char sql_template[] = "INSERT INTO tiles_groups (leader_tile) VALUES(%d) RETURNING id;";

    char sql[128];

    sprintf(sql, sql_template, leader_tile_id);

    res = PQexec(db_info->conn, sql);

    const unsigned int group_id = atoi(PQgetvalue(res, 0, 0));

    PQclear(res);

    return group_id;
}

unsigned int create_virtual_group(DbInfo* db_info) {
    return db_info->last_virtual_group_id++;
}

void read_colors(const DbInfo* const db_info, unsigned int** colors, unsigned int* count) {
    PGresult* res;

    res = PQexec(db_info->conn, "SELECT color FROM (SELECT color, SUM(repeat_count) FROM tile_color GROUP BY 1 ORDER BY 2 DESC) AS t;");

    (*count) = PQntuples(res);

    (*colors) = malloc(sizeof(unsigned int) * (*count));

    for (unsigned int i = 0; i < (*count); ++i) {
        (*colors)[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

void add_tile_to_group(const DbInfo *const db_info, unsigned int group_id, unsigned int tile_id) {
    return;
    PGresult *res;

    const char sql_template[] = "INSERT INTO tile_group (group_id, tile_id) VALUES(%d, %d);";

    char sql[128];

    sprintf(sql, sql_template, group_id, tile_id);

    res = PQexec(db_info->conn, sql);

    PQclear(res);
}

void delete_group(DbInfo *db_info, unsigned int group_id) {
    return;
    PGresult *res;

    const char sql_template[] = "DELETE FROM tiles_groups WHERE id = %d;";

    char sql[128];

    sprintf(sql, sql_template, group_id);

    res = PQexec(db_info->conn, sql);

    PQclear(res);
}

unsigned int create_persistent_group(const DbInfo* const db_info) {
    PGresult* res = PQexec(db_info->conn, "INSERT INTO persistent_groups DEFAULT VALUES RETURNING id");

    const unsigned int id = atoi(PQgetvalue(res, 0, 0));

    PQclear(res);

    return id;
}

void add_tile_to_persistent_group(const DbInfo* const db_info,
                                 unsigned int tile_id,
                                 unsigned int persistent_group_id) {
    const char sql_template[] = "INSERT INTO persistent_group_tile (group_id, tile_id) VALUES (%d, %d)";

    char sql[strlen(sql_template) + 16];

    sprintf(sql, sql_template, persistent_group_id, tile_id);

    PQclear(PQexec(db_info->conn, sql));
}

void create_new_persistent_group_from_parent(const DbInfo* const db_info,
                                             unsigned int parent_persistent_group,
                                             unsigned int color,
                                             unsigned char has_marker) {
    if(has_marker == DOESNT_HAVE_COLOR) {

    }
}

void delete_db_info(DbInfo* db_info) {
    free(db_info->db_buffer->buffer_str);
    free(db_info->db_buffer);
    free(db_info);
}

void insert_tile_color(const unsigned int tile_id,
                       const unsigned int color,
                       const unsigned int repeat_count,
                       const DbInfo* const db_info) {

    if(db_info->db_buffer->current_offset == 0) {
        const size_t insert_max_len = strlen(sql_template_insert_tile_color) + 8 * 3;

        char sql[insert_max_len];

        sprintf(sql, sql_template_insert_tile_color, tile_id, color, repeat_count);
        const size_t insert_len = strlen(sql);
        strncpy(db_info->db_buffer->buffer_str, sql, insert_len);

        db_info->db_buffer->current_offset += insert_len;
    } else {
        const size_t values_max_len = strlen(sql_template_values_tile_color) + 8 * 3;

        if(db_info->db_buffer->current_offset + values_max_len >= db_info->db_buffer->max_buffer_size - 1) {
            flush_buffer_tiles_colors(db_info);

            const size_t insert_max_len = strlen(sql_template_insert_tile_color) + 8 * 3;

            char sql[insert_max_len];

            sprintf(sql, sql_template_insert_tile_color, tile_id, color, repeat_count);

            const size_t insert_len = strlen(sql);
            strncpy(db_info->db_buffer->buffer_str, sql, insert_len);

            db_info->db_buffer->current_offset = insert_len;
        } else {
            char sql[values_max_len];

            sprintf(sql, sql_template_values_tile_color, tile_id, color, repeat_count);
            const size_t values_len = strlen(sql);
            strncpy(db_info->db_buffer->buffer_str + db_info->db_buffer->current_offset, sql, values_len);

            db_info->db_buffer->current_offset += values_len;
        }
    }
}

void flush_buffer_tiles_colors(const DbInfo* const db_info) {
    if(db_info->db_buffer->current_offset == 0)
        return;

    db_info->db_buffer->buffer_str[db_info->db_buffer->current_offset] = ';';
    db_info->db_buffer->buffer_str[db_info->db_buffer->current_offset + 1] = '\0';

    PQclear(PQexec(db_info->conn, db_info->db_buffer->buffer_str));

    db_info->db_buffer->current_offset = 0;
}
