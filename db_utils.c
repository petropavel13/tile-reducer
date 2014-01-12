#include "db_utils.h"

DbInfo* create_db_info(PGconn* conn, size_t pg_sql_buffer_size) {
    DbInfo* db_info = malloc(sizeof(DbInfo));
    db_info->conn = conn;
    db_info->db_buffer = malloc(sizeof(DbBuffer));
    db_info->db_buffer->current_offset = 0;
    db_info->db_buffer->max_buffer_size = pg_sql_buffer_size;
    db_info->db_buffer->buffer_str = malloc(sizeof(char) * db_info->db_buffer->max_buffer_size);

    return db_info;
}

void materialize_count_equality(const DbInfo* const db_info) {
    materialize_tile_color_count(db_info);
    exec_no_result(db_info, "INSERT INTO count_equality_mv (SELECT left_tile_id, right_tile_id FROM count_equality_view_unordered);");
}

void materialize_tile_color_count(const DbInfo* const db_info) {
    clear_tile_color_count(db_info);
    exec_no_result(db_info, "INSERT INTO tile_color_count_mv (SELECT tile_id, count FROM tile_color_records_count_view);");
}

void materialize_tile_color_count_wo_persistent(const DbInfo* const db_info) {
    clear_tile_color_count(db_info);
    exec_no_result(db_info, materialization_of_tile_color_count_without_persistent);
}

void read_related_tiles_ids(const DbInfo* const db_info,
                            const unsigned int related_tile_id,
                            unsigned int * const ids,
                            unsigned int * const count,
                            const unsigned int max_diff_pixels) {
    char sql[strlen(select_related_tiles_ids_sql_template) + UINT32_MAX_CHARS_IN_STRING * 2];

    sprintf(sql, select_related_tiles_ids_sql_template, related_tile_id, max_diff_pixels);

    PGresult* const res = PQexec(db_info->conn, sql);

    (*count) = PQntuples(res);

    for (unsigned int i = 0; i < (*count); ++i) {
        ids[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

void read_working_set_tiles_ids(const DbInfo* const db_info, unsigned int* const ids, unsigned int* const count) {
    PGresult* const res = PQexec(db_info->conn, "SELECT tile_id FROM working_set;");

    (*count) = PQntuples(res);

    for (unsigned int i = 0; i < (*count); ++i) {
        ids[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

void read_persistent_groups(const DbInfo* const db_info,
                            unsigned int ** const leaders_ids,
                            unsigned int* const count) {
    PGresult* const res = PQexec(db_info->conn, "SELECT DISTINCT leader_tile_id FROM persistent_tiles_groups;");

    (*count) = PQntuples(res);

    *leaders_ids = (unsigned int*)malloc(sizeof(unsigned int) * (*count));

    for (unsigned int i = 0; i < (*count); ++i) {
        (*leaders_ids)[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

unsigned char check_is_table_exist(const DbInfo* const db_info, const char* const table_name) {
    char sql[strlen(exists_template) + strlen(table_name) - STRING_TEMPLATE_SIZE];

    sprintf(sql, exists_template, table_name);

    PGresult *res = PQexec(db_info->conn, sql);

    unsigned char exists = PQgetvalue(res, 0, 0)[0] == 't';

    PQclear(res);

    return exists;
}

void create_tables_if_not_exists(const DbInfo* const db_info) {
    if(check_is_table_exist(db_info, "tiles") == TABLE_DOESNT_EXIST) {
        exec_no_result(db_info, create_table_tiles);
    }

    if(check_is_table_exist(db_info,"tiles_groups") == TABLE_DOESNT_EXIST) {
        exec_no_result(db_info, create_table_tiles_groups);
    }

    if(check_is_table_exist(db_info,"tile_color") == TABLE_DOESNT_EXIST) {
        exec_no_result(db_info, create_table_tile_color);
    }

    if(check_is_table_exist(db_info,"persistent_tiles_groups") == TABLE_DOESNT_EXIST) {
        exec_no_result(db_info, create_table_persistent_tiles_groups);
    }

    if(check_is_table_exist(db_info,"working_set") == TABLE_DOESNT_EXIST) {
        exec_no_result(db_info, create_table_working_set);
    }

    if(check_is_table_exist(db_info,"count_equality_mv") == TABLE_DOESNT_EXIST) {
        exec_no_result(db_info, create_table_count_equality_mv);
    }

    if(check_is_table_exist(db_info,"tile_color_count_mv") == TABLE_DOESNT_EXIST) {
        exec_no_result(db_info, create_tile_color_count_mv_table);
    }


    exec_no_result(db_info, create_tile_color_records_count_view);
    exec_no_result(db_info, create_count_equality_view_unordered);
}

void clear_all_data(const DbInfo* const db_info) {
    exec_no_result(db_info, clear_data_sql);
}

void clear_session_data(const DbInfo* const db_info) {
    exec_no_result(db_info, clear_session_data_sql);
}

void write_tiles_paths(const DbInfo* const db_info,
                                    char **const paths,
                                    unsigned int total_count,
                                    unsigned int *const ids_in_pg,
                                    void (*callback) (unsigned int)) {
    PGresult *res = NULL;

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

    size_t insert_len = insert_template_len + strlen(paths[0]) - STRING_TEMPLATE_SIZE;
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

            insert_len = insert_template_len + strlen(paths[i]) - STRING_TEMPLATE_SIZE;

            sprintf(sql_insert, sql_template_insert, paths[i]);
            strncpy(db_info->db_buffer->buffer_str, sql_insert, insert_len);

            db_info->db_buffer->current_offset = insert_len;
        } else {
            sprintf(sql_values, sql_template_values, paths[i]);

            values_len = values_template_len + strlen(paths[i]) - STRING_TEMPLATE_SIZE;

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
}

void read_tiles_ids(const DbInfo *const db_info, unsigned int* const ids_in_pg) {
    const char select_ids_sql[] = "SELECT id FROM tiles;";

    PGresult *res = PQexec(db_info->conn, select_ids_sql);

    const unsigned int count_objects = PQntuples(res);

    for (unsigned int i = 0; i < count_objects; ++i) {
        ids_in_pg[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

unsigned char check_tiles_in_db(const DbInfo* const db_info, unsigned int guess_count) {
    const char count_sql[] = "SELECT COUNT(*) FROM tiles;";

    PGresult *res = PQexec(db_info->conn, count_sql);

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


void add_tile_to_group(const DbInfo * const db_info, unsigned int leader_tile_id, unsigned int tile_id) {
    char sql[strlen(sql_template_add_tile_to_group) + UINT32_MAX_CHARS_IN_STRING * 2];

    sprintf(sql, sql_template_add_tile_to_group, leader_tile_id, tile_id);

    exec_no_result(db_info, sql);
}

void add_tile_to_group_using_buffer(const DbInfo * const db_info, unsigned int leader_tile_id, unsigned int tile_id) {
    if(get_buffer_status(db_info, 0) == BUFFER_EMPTY) {
        const size_t sql_insert_max_size = strlen(sql_buffer_template_insert_tile_to_group) + UINT32_MAX_CHARS_IN_STRING * 2;

        char sql[sql_insert_max_size];
        sprintf(sql, sql_buffer_template_insert_tile_to_group, leader_tile_id, tile_id);

        add_to_buffer(db_info, sql, 0);
    } else {
        const size_t sql_values_max_size = strlen(sql_buffer_template_values_tile_to_group) + UINT32_MAX_CHARS_IN_STRING * 2;

        if(get_buffer_status(db_info, sql_values_max_size) == BUFFER_FULL) {
            flush_db_buffer(db_info);

            add_tile_to_persistent_group_using_buffer(db_info, tile_id, leader_tile_id); // recursion
        } else {
            char sql[sql_values_max_size];
            sprintf(sql, sql_buffer_template_values_tile_to_group, leader_tile_id, tile_id);

            add_to_buffer(db_info, sql, db_info->db_buffer->current_offset);
        }
    }
}

void load_zero_equals_ids_leaders(const DbInfo* const db_info, unsigned int* ids_in_pg, unsigned int *count) {
    PGresult* res = PQexec(db_info->conn, "SELECT DISTINCT left_tile_id FROM count_equality_mv;");

    *count = PQntuples(res);

    for (unsigned int i = 0; i < *count; ++i) {
        ids_in_pg[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

void load_zero_equals_ids_for_tile(const DbInfo* const db_info,
                          unsigned int tile_id,
                          unsigned int* ids_in_pg,
                          unsigned int* count) {
    const char sql_template[] = "SELECT right_tile_id FROM count_equality_mv WHERE left_tile_id = %u;";

    char sql[strlen(sql_template) + UINT32_MAX_CHARS_IN_STRING];

    sprintf(sql, sql_template, tile_id);

    PGresult* res = PQexec(db_info->conn, sql);

    *count = PQntuples(res);

    for (unsigned int i = 0; i < *count; ++i) {
        ids_in_pg[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

void add_tile_to_persistent_group(const DbInfo* const db_info,
                                const unsigned int tile_id,
                                const unsigned int leader_tile_id) {
    char sql[strlen(sql_template_insert_persistent_group_tile) + UINT32_MAX_CHARS_IN_STRING * 2];

    sprintf(sql, sql_template_insert_persistent_group_tile, leader_tile_id, tile_id);
    exec_no_result(db_info, sql);
}

void add_tile_to_persistent_group_using_buffer(const DbInfo* const db_info,
                                        const unsigned int tile_id,
                                        const unsigned int leader_tile_id) {
    if(get_buffer_status(db_info, 0) == BUFFER_EMPTY) {
        const size_t sql_insert_max_size = strlen(sql_buffer_template_insert_persistent_group_tile) + UINT32_MAX_CHARS_IN_STRING * 2;

        char sql[sql_insert_max_size];
        sprintf(sql, sql_buffer_template_insert_persistent_group_tile, leader_tile_id, tile_id);

        add_to_buffer(db_info, sql, 0);
    } else {
        const size_t sql_values_max_size = strlen(sql_template_values_persistent_group_tile) + UINT32_MAX_CHARS_IN_STRING * 2;

        if(get_buffer_status(db_info, sql_values_max_size) == BUFFER_FULL) {
            flush_db_buffer(db_info);

            add_tile_to_persistent_group_using_buffer(db_info, tile_id, leader_tile_id); // recursion
        } else {
            char sql[sql_values_max_size];
            sprintf(sql, sql_template_values_persistent_group_tile, leader_tile_id, tile_id);

            add_to_buffer(db_info, sql, db_info->db_buffer->current_offset);
        }
    }
}

void reduce_persistent_tiles_groups(const DbInfo* const db_info) {
    exec_no_result(db_info, reduce_persistent_tiles_groups_sql);
}

void destroy_db_info(DbInfo *const db_info) {
    free(db_info->db_buffer->buffer_str);
    free(db_info->db_buffer);
    PQfinish(db_info->conn);
    free(db_info);
}

void add_to_buffer(const DbInfo* const db_info, const char * const sql, const size_t offset) {
    const size_t len = strlen(sql);
    strncpy(db_info->db_buffer->buffer_str + offset, sql, len);

    db_info->db_buffer->current_offset += len;
}

unsigned char get_buffer_status(const DbInfo* const db_info, const size_t chars_to_insert) {
    if(db_info->db_buffer->current_offset == 0) {
        return BUFFER_EMPTY;
    } else if(db_info->db_buffer->current_offset + chars_to_insert >= db_info->db_buffer->max_buffer_size) {
        return BUFFER_FULL;
    }

    return BUFFER_OK;
}

void insert_tile_color_using_buffer(const unsigned int tile_id,
                                    const unsigned int color,
                                    const unsigned int repeat_count,
                                    const DbInfo* const db_info) {

    if(get_buffer_status(db_info, 0) == BUFFER_EMPTY) {
        const size_t sql_insert_max_size = strlen(sql_template_insert_tile_color) + UINT32_MAX_CHARS_IN_STRING * 3;

        char sql[sql_insert_max_size];
        sprintf(sql, sql_template_insert_tile_color, tile_id, color, repeat_count);

        add_to_buffer(db_info, sql, 0);
    } else {
        const size_t sql_values_max_size = strlen(sql_template_values_tile_color) + UINT32_MAX_CHARS_IN_STRING * 3;

        if(get_buffer_status(db_info, sql_values_max_size) == BUFFER_FULL) {
            flush_db_buffer(db_info);

            insert_tile_color_using_buffer(tile_id, color, repeat_count, db_info); // recursion
        } else {
            char sql[sql_values_max_size];
            sprintf(sql, sql_template_values_tile_color, tile_id, color, repeat_count);

            add_to_buffer(db_info, sql, db_info->db_buffer->current_offset);
        }
    }
}

void flush_db_buffer(const DbInfo* const db_info) {
    if(db_info->db_buffer->current_offset == 0)
        return;

    db_info->db_buffer->buffer_str[db_info->db_buffer->current_offset] = ';';
    db_info->db_buffer->buffer_str[db_info->db_buffer->current_offset + 1] = '\0';

    exec_no_result(db_info, db_info->db_buffer->buffer_str);

    db_info->db_buffer->current_offset = 0;
}

void create_working_set(const DbInfo* const db_info) {
    exec_no_result(db_info, "INSERT INTO working_set(tile_id) SELECT id FROM tiles;");
}

void create_working_set_wo_persistent_records_w_max_diff(const DbInfo* const db_info, const unsigned short max_diff_pixels) {
    char sql[strlen(insert_into_working_set_without_persistent_wth_max_diff_sql_template) + UINT32_MAX_CHARS_IN_STRING];

    sprintf(sql, insert_into_working_set_without_persistent_wth_max_diff_sql_template, max_diff_pixels);
    exec_no_result(db_info, sql);
}

unsigned int get_next_tile_from_working_set(const DbInfo* const db_info) {
    PGresult* res = PQexec(db_info->conn, "SELECT tile_id FROM working_set LIMIT 1;");

    unsigned int id = 0;

    if(PQntuples(res) > 0) {
        id = atoi(PQgetvalue(res, 0, 0));
    }

    PQclear(res);

    return id;
}
