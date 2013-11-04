#include "cluster_utils.h"

void make_persistent_groups(DbInfo* const db_info,
                            GenericNode *const tiles_root_node,
                            unsigned int total,
                            CacheInfo* const cache_info) {
    clear_working_set(db_info);
    create_working_set(db_info);

    unsigned int persistent_group_id = PERSISTENT_GROUP_NOT_DEFINED;

    unsigned int next_tile_id = get_next_tile_from_working_set(db_info);
    remove_tile_from_working_set(db_info, next_tile_id);

    unsigned int* pg_ids = malloc(sizeof(unsigned int) * total);
    unsigned int temp_count;

    Tile* next_tile = NULL;
    const Tile** const equal_candidates = malloc(sizeof(Tile*) * total); // prevent many malloc/free calls

    unsigned int candidate_tile_id = 0;

    unsigned short int* const results = malloc(sizeof(unsigned short int) * total); // prevent many malloc/free calls

    unsigned int diff_result = 0;

    while(next_tile_id > 0) {
        printf("current: %d\n", next_tile_id);
        fflush(stdout);
        next_tile = (Tile*) find(tiles_root_node, next_tile_id)->data;
        load_zero_equals_ids(db_info, next_tile_id, pg_ids, &temp_count);

        if(temp_count > 1) {
            for (unsigned int i = 0; i < temp_count; ++i) {
                equal_candidates[i] = (Tile*) find(tiles_root_node, pg_ids[i])->data;
            }

            calc_diff_one_with_many(next_tile, equal_candidates, temp_count, cache_info, results);

            for (unsigned int i = 0; i < temp_count; ++i) {
                diff_result = results[i];
                candidate_tile_id = pg_ids[i];

                if(diff_result == 0) {
                    if(persistent_group_id == PERSISTENT_GROUP_NOT_DEFINED) {
                        persistent_group_id = create_persistent_group(db_info, next_tile_id);
                    }

                    add_tile_to_persistent_group_using_buffer(db_info, candidate_tile_id, persistent_group_id);
                }
            }

            flush_db_buffer(db_info);
        } else if(temp_count == 1) {
            equal_candidates[0] = (Tile*) find(tiles_root_node, pg_ids[0])->data;
            diff_result = calc_diff(next_tile, equal_candidates[0], cache_info);

            if(diff_result == 0) {
                candidate_tile_id = pg_ids[0];

                persistent_group_id = create_persistent_group(db_info, next_tile_id);
                add_tile_to_persistent_group(db_info, candidate_tile_id, persistent_group_id);
            }
        }

        remove_tiles_from_working_set_via_zero_equals(db_info, next_tile_id);
        remove_tile_from_working_set(db_info, next_tile_id);

        next_tile_id = get_next_tile_from_working_set(db_info);
        persistent_group_id = PERSISTENT_GROUP_NOT_DEFINED;
    }

    free(pg_ids);
    free(equal_candidates);
    free(results);
}

void clusterize(GroupElement* const tiles_sequence,
                unsigned int total,
                unsigned int max_diff_pixels,
                unsigned int max_allowed_tiles,
                CacheInfo *const cache_info,
                DbInfo* const db_info) {
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

    PathPoint* left_path = make_group(tiles_sequence, total, 1, root_path, cache_info, db_info, max_diff_pixels, max_allowed_tiles, 1);
    PathPoint* right_path = make_group(tiles_sequence, total, 0, root_path, cache_info, db_info, max_diff_pixels, max_allowed_tiles, 1);

    const char best_select = choose_best(left_path, right_path);

    if(best_select == -1) {
        delete_path(right_path, db_info);
    } else if(best_select == 1) {
        delete_path(left_path, db_info);
    } else if(best_select == 0) {
        delete_path(left_path, db_info); // or right
    }
}

PathPoint* make_group(GroupElement* const rest_tiles,
                      unsigned int total,
                      unsigned int offset,
                      PathPoint *const parent_path,
                      CacheInfo *const cache_info,
                      DbInfo* const db_info,
                      unsigned int max_diff_pixels,
                      unsigned int max_allowed_tiles,
                      unsigned int max_allowed_groups) {
    const GroupElement* temp = get_element_with_index(rest_tiles->first, offset);

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


    PathPoint* left_path = make_group(rest_tiles, total, ++offset, child_left, cache_info, db_info, max_diff_pixels, max_allowed_tiles, max_allowed_groups);


    GroupElement* current_rest = malloc(sizeof(GroupElement));
    current_rest->first = current_rest;
    GroupElement* current_rest_prev = NULL;

    Tile* const leader_tile = temp->node;

    const unsigned int group_id = create_virtual_group(db_info/*, leader_tile->tile_id, leader_tile->tile_id*/);

    unsigned short int diff;

    unsigned char res;

    unsigned int current_rest_count = 0;

    unsigned long key;

    while ((temp = temp->next) != NULL) {
        diff = 0;

        key = make_key(leader_tile->tile_id, temp->node->tile_id);

        res = get_diff_from_cache(key, cache_info, &diff);

        if(res == CACHE_HIT) {
            //
        } else if(res == CACHE_MISS) {
            diff = calc_diff(leader_tile, temp->node, cache_info);
            push_edge_to_cache(key, diff, cache_info);
        }

        if(diff <= max_diff_pixels) {
            add_tile_to_group(db_info, group_id, temp->node->tile_id);
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
        right_path = make_group(current_rest, total, 0, child_right, cache_info, db_info, max_diff_pixels, max_allowed_tiles, max_allowed_groups);
    }

    const char best_select = choose_best(left_path, right_path);

    if(best_select == -1) {
        delete_path(right_path, db_info);

        parent_path->child_left = child_left;

        return left_path;
    } else if(best_select == 1) {
        delete_path(left_path, db_info);

        parent_path->child_right = child_right;

        return right_path;
    } else if(best_select == 0) {
        delete_path(left_path, db_info); // or right

        parent_path->child_right = child_right;

        return right_path;
    }

    return NULL; // impossible
}

void delete_path(PathPoint* point, DbInfo* const db_info) {
    if(point->selected_point != NULL) {
        GroupElement* sequence_head = point->selected_point->sequence_head == NULL ? NULL : point->selected_point->sequence_head->first;
        GroupElement* prev = NULL;

        while (sequence_head != NULL) {
            prev = sequence_head;
            sequence_head = sequence_head->next;
            free(prev);
        }

        delete_group(db_info, point->selected_point->group_id);

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

GroupElement* get_element_with_index(GroupElement* const search_start_element, unsigned int index) {
    unsigned int current = 0;
    GroupElement* temp = search_start_element;

    while ((current++ < index) && (temp != NULL)) {
        temp = temp->next;
    }

    return temp;
}

GroupElement* find_tile_with_id(GroupElement* const search_start_element, unsigned int tile_id) {
    GroupElement* temp = search_start_element;

    while (temp!= NULL) {
        if(temp->node->tile_id == tile_id) {
            return temp;
        }

        temp = temp->next;
    }

    return NULL;
}

SelectedPoint* get_selected_point_for_branch(const PathPoint* branch) {
    SelectedPoint* sp = branch->selected_point;

    const PathPoint* temp = branch;

    while(temp != NULL && sp == NULL) {
        sp = temp->selected_point;
        temp = temp->parent;
    }

    return sp;
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
