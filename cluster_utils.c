#include "cluster_utils.h"

void clusterize(GroupElement* const tiles_sequence,
                unsigned int total,
                unsigned int max_diff_pixels,
                unsigned int max_allowed_tiles,
                CacheInfo *const cache_info,
                PGconn *conn) {
    SelectedPoint* const root_selected_point = malloc(sizeof(SelectedPoint));
    root_selected_point->groups_count = 0;
    root_selected_point->group_id = 0;
    root_selected_point->leader_node = tiles_sequence->first->node;
    root_selected_point->rest_count = total;
    root_selected_point->sequence_head = tiles_sequence->first;

    PathPoint* const root_path = malloc(sizeof(PathPoint));
    root_path->parent = NULL;
    root_path->child_left = NULL;
    root_path->child_right = NULL;
    root_path->selected_point = root_selected_point;

    PathPoint* left_path = make_group(tiles_sequence, total, 1, root_path, cache_info, conn, max_diff_pixels, max_allowed_tiles, 1);
    PathPoint* right_path = make_group(tiles_sequence, total, 0, root_path, cache_info, conn, max_diff_pixels, max_allowed_tiles, 1);

    const char best_select = choose_best(left_path, right_path);

    if(best_select == -1) {
        delete_path(right_path, conn);
    } else if(best_select == 1) {
        delete_path(left_path, conn);
    } else if(best_select == 0) {
        delete_path(left_path, conn); // or right
    }
}

PathPoint* make_group(GroupElement* const rest_tiles,
                      unsigned int total,
                      unsigned int offset,
                      PathPoint *const parent_path,
                      CacheInfo *const cache_info,
                      PGconn *conn,
                      unsigned int max_diff_pixels,
                      unsigned int max_allowed_tiles,
                      unsigned int max_allowed_groups) {
    const GroupElement* temp = get_element_with_index(rest_tiles, offset);

    if(temp == NULL) {
        return parent_path;
    }

    const SelectedPoint* const parent_sp = get_selected_point_for_branch(parent_path);

    if((total - (parent_sp->groups_count + parent_sp->rest_count)) > max_allowed_tiles) {
        return parent_path;
    }

    PathPoint* const child_left = malloc(sizeof(PathPoint));
    child_left->parent = parent_path;
    child_left->child_left = NULL;
    child_left->child_right = NULL;
    child_left->selected_point = NULL;


    PathPoint* left_path = make_group(rest_tiles, total, ++offset, child_left, cache_info, conn, max_diff_pixels, max_allowed_tiles, max_allowed_groups);


    GroupElement* current_rest = malloc(sizeof(GroupElement));
    current_rest->first = current_rest;
    GroupElement* current_rest_prev = NULL;

    Tile* const leader_tile = temp->node;

    const unsigned int group_id = create_group(conn, leader_tile->tile_id, leader_tile->tile_id);

    unsigned short int diff;

    unsigned char res;

    unsigned int current_rest_count = 0;

    unsigned long key;

    while ((temp = temp->next) != NULL) {
        diff = 0;

        key = make_key(leader_tile->tile_id, temp->node->tile_id);

        res = get_diff(key, cache_info, &diff);

        if(res == CACHE_HIT) {
            //
        } else if(res == CACHE_MISS) {
            diff = calc_diff(leader_tile, temp->node, cache_info);
            push_edge_to_cache(key, diff, cache_info);
        }

        if(diff <= max_diff_pixels) {
            add_tile_to_group(conn, group_id, temp->node->tile_id);
        } else {
            current_rest->node = temp->node;

            current_rest->next = malloc(sizeof(GroupElement));
            current_rest->next->first = current_rest->first;

            current_rest_prev = current_rest;

            current_rest = current_rest->next;

            current_rest_count++;
        }
    }

    SelectedPoint* const right_selected_point = malloc(sizeof(SelectedPoint));
    right_selected_point->groups_count = parent_sp->groups_count + 1;
    right_selected_point->group_id = group_id;
    right_selected_point->leader_node = leader_tile;


    if(current_rest_count == 0) {
        free(current_rest);

        right_selected_point->rest_count = parent_sp->rest_count - 1;
        right_selected_point->sequence_head = NULL;

    } else {
        free(current_rest);
        current_rest_prev->next = NULL;
        current_rest = current_rest_prev;

        right_selected_point->rest_count = parent_sp->rest_count - (get_count_of_sequence(rest_tiles) - offset - current_rest_count) - 1;
        right_selected_point->sequence_head = current_rest;
    }


    PathPoint* child_right = malloc(sizeof(PathPoint));
    child_right->parent = parent_path;
    child_right->child_left = NULL;
    child_right->child_right = NULL;
    child_right->selected_point = right_selected_point;


    PathPoint* right_path = child_right;

    if(right_selected_point->groups_count <= max_allowed_groups) {
        right_path = make_group(current_rest, total, 0, child_right, cache_info, conn, max_diff_pixels, max_allowed_tiles, max_allowed_groups);
    }

    const char best_select = choose_best(left_path, right_path);

    if(best_select == -1) {
        delete_path(right_path, conn);

        parent_path->child_left = child_left;

        return left_path;
    } else if(best_select == 1) {
        delete_path(left_path, conn);

        parent_path->child_right = child_right;

        return right_path;
    } else if(best_select == 0) {
        delete_path(left_path, conn); // or right

        parent_path->child_right = child_right;

        return right_path;
    }

    return NULL; // impossible
}

void delete_path(PathPoint* point, PGconn* conn) {
    if(point->selected_point != NULL) {
        GroupElement* sequence_head = point->selected_point->sequence_head == NULL ? NULL : point->selected_point->sequence_head->first;
        GroupElement* prev = NULL;

        while (sequence_head != NULL) {
            prev = sequence_head;
            sequence_head = sequence_head->next;
            free(prev);
        }

        delete_group(conn, point->selected_point->group_id);

        free(point->selected_point);
    }

    free(point);
}

unsigned int get_count_of_sequence(const GroupElement* const head) {
    unsigned int count = 0;

    const GroupElement* temp = head->first;

    while (temp != NULL) {
        count++;

        temp = temp->next;
    }

    return count;
}


char choose_best(const PathPoint* left_path, const PathPoint* right_path) {
    const SelectedPoint* lsp = get_selected_point_for_branch(left_path);
    const SelectedPoint* rsp = get_selected_point_for_branch(right_path);

#ifdef DEBUG
    if(left_selected_point == NULL || right_selected_point == NULL) {
        printf("\n\n --Can't compare Paths-- \n\n");
    }
#endif

    const unsigned int l_rating = lsp->groups_count + lsp->rest_count;
    const unsigned int r_rating = rsp->groups_count + rsp->rest_count;

    return (-1) * (l_rating < r_rating) + 1 * (l_rating > r_rating);
}
