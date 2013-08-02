#include "db_utils.h"

void create_tables_if_not_exists(PGconn* conn) {
    const char exists_template[] = "SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name='%s')";

    char sql[512];

    PGresult *res;

    sprintf(sql, exists_template, "tiles");

    res = PQexec(conn, sql);

    unsigned char exists = PQgetvalue(res, 0, 0)[0] == 't';

    PQclear(res);

    if(!exists) {

        const char create_table_tiles[] = "CREATE TABLE tiles\
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

        res = PQexec(conn, create_table_tiles);

        PQclear(res);
    }


    sprintf(sql, exists_template, "tiles_groups");

    res = PQexec(conn, sql);

    exists = PQgetvalue(res, 0, 0)[0] == 't';

    PQclear(res);

    if(!exists) {

        const char create_table_group[] = "CREATE TABLE tiles_groups\
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

        res = PQexec(conn, create_table_group);

        PQclear(res);
    }

    sprintf(sql, exists_template, "tile_group");

    res = PQexec(conn, sql);

    exists = PQgetvalue(res, 0, 0)[0] == 't';

    PQclear(res);

    if(!exists) {
        const char create_table_tile_group[] = "CREATE TABLE tile_group\
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

        res = PQexec(conn, create_table_tile_group);

        PQclear(res);
    }

}

void clear_all_data(PGconn *conn) {
    const char clear_data_sql[] = "TRUNCATE tile_group, tiles_groups, tile_tile_diff, tiles;";

    PGresult *res;

    res = PQexec(conn, clear_data_sql);

    PQclear(res);
}

void write_tiles_paths_to_pg(PGconn* conn,
                             char **const paths,
                             unsigned int total_count,
                             unsigned int *const ids_in_pg,
                             void (*callback) (unsigned int, unsigned int, const char*)) {
    PGresult *res;

    res = PQexec(conn, "BEGIN;");
    PQclear(res);

    const char sql_template[] = "INSERT INTO tiles (tile_path) VALUES('%s') RETURNING id;";

    char sql[128];

    for (unsigned int i = 0; i < total_count; ++i)
    {
        sprintf(sql, sql_template, paths[i]);

        if(callback != NULL) {
            callback(i, total_count, sql);
        }

        res = PQexec(conn, sql);

        ids_in_pg[i] = atoi(PQgetvalue(res, 0, 0));

        PQclear(res);
        //        memset (sql, 0, 128);
    }

    res = PQexec(conn, "COMMIT;");
    PQclear(res);
}

void read_tiles_ids(PGconn* conn, unsigned int *const ids_in_pg) {
    const char select_ids_sql[] = "SELECT id FROM tiles;";

    PGresult *res;

    res = PQexec(conn, select_ids_sql);

    const unsigned int count_objects = PQntuples(res);

    for (unsigned int i = 0; i < count_objects; ++i) {
        ids_in_pg[i] = atoi(PQgetvalue(res, i, 0));
    }

    PQclear(res);
}

unsigned int check_tiles_in_db(PGconn* conn, unsigned int guess_count) {
    const char count_sql[] = "SELECT COUNT(*) FROM tiles;";

    PGresult *res;

    res = PQexec(conn, count_sql);

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


unsigned int create_group(PGconn *conn, unsigned int leader_tile_id, unsigned int node_id) {
    return 42;
    PGresult *res;

    const char sql_template[] = "INSERT INTO tiles_groups (leader_tile, node_id) VALUES(%d, %d) RETURNING id;";

    char sql[128];

    sprintf(sql, sql_template, leader_tile_id, node_id);

    res = PQexec(conn, sql);

    const unsigned int group_id = atoi(PQgetvalue(res, 0, 0));

    PQclear(res);

    return group_id;
}

void add_tile_to_group(PGconn *conn, unsigned int group_id, unsigned int tile_id) {
    return;
    PGresult *res;

    const char sql_template[] = "INSERT INTO tile_group (group_id, tile_id) VALUES(%d, %d);";

    char sql[128];

    sprintf(sql, sql_template, group_id, tile_id);

    res = PQexec(conn, sql);

    PQclear(res);
}

void delete_group(PGconn *conn, unsigned int group_id) {
    return;
    PGresult *res;

    const char sql_template[] = "DELETE FROM tiles_groups WHERE id = %d;";

    char sql[128];

    sprintf(sql, sql_template, group_id);

    res = PQexec(conn, sql);

    PQclear(res);
}
