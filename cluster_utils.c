#include "cluster_utils.h"

void make_persistent_groups(const DbInfo *const db_info,
                            GenericNode *const tiles_root_node,
                            const unsigned int total,
                            CacheInfo* const cache_info,
                            void (*callback)(unsigned int, unsigned int)) {
    TreeInfo* const tree_info = (TreeInfo*)malloc(sizeof(TreeInfo));
    tree_info->data_destructor = NULL;

    unsigned int* pg_ids = malloc(sizeof(unsigned int) * total);
    unsigned int t_ids_count;

    load_zero_equals_ids_leaders(db_info, pg_ids, &t_ids_count);

    GenericNode* not_used = create_tiles_tree_from_tiles_ids(tiles_root_node, pg_ids, t_ids_count);

    Tile* next_tile = NULL;
    const Tile** const equal_candidates = (const Tile**)malloc(sizeof(Tile*) * total); // prevent many malloc/free calls

    unsigned short int* const results = malloc(sizeof(unsigned short int) * total); // prevent many malloc/free calls

    unsigned int rest_count = t_ids_count;

    while(not_used != NULL) {
        next_tile = (Tile*) not_used->data;

        load_zero_equals_ids_for_tile(db_info, next_tile->tile_id, pg_ids, &t_ids_count);

        if(callback != NULL) {
            callback(rest_count, t_ids_count);
        }

        for (unsigned int i = 0; i < t_ids_count; ++i) {
            equal_candidates[i] = (Tile*) find(tiles_root_node, pg_ids[i])->data;
        }

        if(t_ids_count > 1) {
            calc_diff_one_with_many(next_tile, equal_candidates, t_ids_count, cache_info, results);

            for (unsigned int i = 0; i < t_ids_count; ++i) {
                if(results[i] == 0) {
                    add_tile_to_persistent_group_using_buffer(db_info, equal_candidates[i]->tile_id, next_tile->tile_id);

                    if(callback != NULL) {
                        if(find(not_used, equal_candidates[i]->tile_id) != NULL) {
                            rest_count--;
                        }
                    }
                    not_used = remove_node(not_used, equal_candidates[i]->tile_id, tree_info);

                }
            }

            flush_db_buffer(db_info);
        } else if(t_ids_count == 1) {  // compare 1 to 1 (use simpler functions)
            if(calc_diff(next_tile, equal_candidates[0], cache_info) == 0) {
                add_tile_to_persistent_group(db_info, equal_candidates[0]->tile_id, next_tile->tile_id);

                if(callback != NULL) {
                    if(find(not_used, equal_candidates[0]->tile_id) != NULL) {
                        rest_count--;
                    }
                }

                not_used = remove_node(not_used, equal_candidates[0]->tile_id, tree_info);
            }
        }

        not_used = remove_node(not_used, next_tile->tile_id, tree_info);

        rest_count--;

        t_ids_count = 0;
    }

    free(pg_ids);
    free(equal_candidates);
    free(results);

    destroy_tree(not_used, tree_info);
    free(tree_info);
}

void migrate_tile(GenericNode** from, GenericNode** to, const unsigned int tile_id) {
    TreeInfo* const tree_info = (TreeInfo*)malloc(sizeof(TreeInfo));
    tree_info->data_destructor = NULL;

    *to = insert(*to, tile_id, find(*from, tile_id)->data);
    *from = remove_node(*from, tile_id, tree_info);

    free(tree_info);
}

void delete_tile_groups_sequence(TileGroupsSequence* const sequence) {
    if(sequence == NULL)
        return;

    delete_tile_groups_sequence(sequence->next);

    free(sequence);
}

void delete_binded_tiles_sequence(BindedTilesSequence* const sequence) {
    if(sequence == NULL)
        return;

    delete_binded_tiles_sequence(sequence->next);
    sequence->next = NULL;

    if(sequence->item != NULL) {
        free(sequence->item);
        sequence->item = NULL;
    }

    free(sequence);
}

void delete_result_slot(ResultSlot slot) {
    delete_tile_groups_sequence(slot.tg_seq);
    delete_binded_tiles_sequence(slot.bt_seq);

    slot.is_empty = 1;
}

void clusterize(GenericNode* const all_tiles,
                const unsigned int all_tiles_count,
                const unsigned int max_diff_pixels,
                const DbInfo* const db_info,
                CacheInfo* const cache_info) {
    materialize_tile_color_count_wo_persistent(db_info);

    clear_working_set(db_info);
    create_working_set_wo_persistent_records_w_max_diff(db_info, max_diff_pixels);

    unsigned int working_set_count = 0;
    unsigned int* const working_set_ids = (unsigned int*)malloc(sizeof(unsigned int) * all_tiles_count);

    read_working_set_tiles_ids(db_info, working_set_ids, &working_set_count);

    GenericNode* free_tiles = create_tiles_tree_from_tiles_ids(all_tiles, working_set_ids, working_set_count);

    free(working_set_ids);

    GenericNode* used_tiles = create_node(0, NULL); // dummy

    unsigned int* persistent_groups_ids = NULL;
    unsigned int* persistent_groups_leaders_ids = NULL;
    unsigned int persistent_groups_count = 0;

    read_persistent_groups(db_info, &persistent_groups_ids, &persistent_groups_leaders_ids, &persistent_groups_count);

    TileGroupsSequence* const tile_group_sequence = create_tiles_groups_sequence_from_ids(all_tiles, persistent_groups_ids, persistent_groups_leaders_ids,persistent_groups_count);

    BindedTilesSequence* const binded_tiles_sequence = (BindedTilesSequence*)malloc(sizeof(BindedTilesSequence));
    binded_tiles_sequence->item = NULL;
    binded_tiles_sequence->next = NULL;

    TilesSequence* t_tiles_sequence_from_free = (TilesSequence*)malloc(sizeof(TilesSequence));
    t_tiles_sequence_from_free->tile = NULL;
    t_tiles_sequence_from_free->next = NULL;

    TilesSequence* t_last = make_tile_sequence_from_tree(free_tiles, t_tiles_sequence_from_free);

    TilesSequence* t_tiles_sequence = t_tiles_sequence_from_free;

    TreeInfo* const tree_info_sequence = (TreeInfo*)malloc(sizeof(TreeInfo));
    tree_info_sequence->data_destructor = &tile_sequence_destructor;

    TreeInfo* tree_info_tile = (TreeInfo*)malloc(sizeof(TreeInfo));
    tree_info_tile->data_destructor = NULL;

    GenericNode* related_tiles = NULL;

    unsigned int* const related_ids = (unsigned int*)malloc(sizeof(unsigned int) * working_set_count);

    unsigned int related_count = 0;

    TilesSequence* t_related_sequence = NULL;

    while (t_tiles_sequence != NULL && t_tiles_sequence != t_last) {
        read_related_tiles_ids(db_info, t_tiles_sequence->tile->tile_id, related_ids, &related_count, max_diff_pixels);

        t_related_sequence = create_tiles_sequence_from_tile_ids(free_tiles, related_ids, related_count);

        if(t_related_sequence == NULL) {
            working_set_count--;
            free_tiles = remove_node(free_tiles, t_tiles_sequence->tile->tile_id, tree_info_tile);
        } else {
            clean_related_group(t_tiles_sequence->tile, &t_related_sequence, related_count, max_diff_pixels, cache_info);
            related_tiles = insert(related_tiles, t_tiles_sequence->tile->tile_id, t_related_sequence);
        }

        t_tiles_sequence = t_tiles_sequence->next;
    }

    delete_tiles_sequence(t_tiles_sequence_from_free);

    t_tiles_sequence_from_free = (TilesSequence*)malloc(sizeof(TilesSequence));
    t_tiles_sequence_from_free->tile = NULL;
    t_tiles_sequence_from_free->next = NULL;

    t_last = make_tile_sequence_from_tree(free_tiles, t_tiles_sequence_from_free);

    t_tiles_sequence = t_tiles_sequence_from_free;

    while (t_tiles_sequence != NULL && t_tiles_sequence != t_last) {
        t_related_sequence = (TilesSequence*) find(related_tiles, t_tiles_sequence->tile->tile_id)->data;

        if(t_related_sequence == NULL) {
            working_set_count--;
            free_tiles = remove_node(free_tiles, t_tiles_sequence->tile->tile_id, tree_info_tile);
            related_tiles = remove_node(related_tiles, t_tiles_sequence->tile->tile_id, tree_info_sequence);
        }

        t_tiles_sequence = t_tiles_sequence->next;
    }

    unsigned long f_count = 0;
    calc_elements_count(free_tiles, &f_count);

    free(related_ids);

//    const NodeResult best_result = calc_node_result(tile_group_sequence,
//                                                    persistent_groups_count,
//                                                    binded_tiles_sequence,
//                                                    &used_tiles,
//                                                    &free_tiles,
//                                                    working_set_count,
//                                                    (Tile*) free_tiles->data,
//                                                    related_tiles,
//                                                    max_diff_pixels,
//                                                    cache_info);

    hungry_by_groups(tile_group_sequence, binded_tiles_sequence, &used_tiles, &free_tiles, (Tile*) free_tiles->data, max_diff_pixels, cache_info);

//const unsigned int best_by_groupss = hungry_by_groups(tile_group_sequence,
//                                                      persistent_groups_count,
//                                                      binded_tiles_sequence,
//                                                      &used_tiles,
//                                                      &free_tiles,
//                                                      (Tile*) free_tiles->data,
//                                                      related_tiles,
//                                                      max_diff_pixels,
//                                                      cache_info);

    delete_tile_groups_sequence(tile_group_sequence);
    delete_binded_tiles_sequence(binded_tiles_sequence);

    destroy_tree(free_tiles, tree_info_sequence);
    destroy_tree(used_tiles, tree_info_sequence);
    destroy_tree(related_tiles, tree_info_sequence);

    free(tree_info_sequence);

    delete_tiles_sequence(t_tiles_sequence_from_free);
}

void clean_related_group(const Tile* const tile,
                         TilesSequence** related_sequence_for_tile,
                         const unsigned int related_sequence_count,
                         const unsigned short int max_diff_pixels,
                         CacheInfo* const cache_info) {
    if(related_sequence_count > 1) {
        Tile** const related_tiles_for_tile = (Tile**)malloc(sizeof(Tile*) * related_sequence_count);

        TilesSequence* t_related_tile = *related_sequence_for_tile;

        for (unsigned int i = 0; i < related_sequence_count; ++i) {
            related_tiles_for_tile[i] = t_related_tile->tile;
            t_related_tile = t_related_tile->next;
        }

        unsigned short int* const diff_results = (unsigned short int*)malloc(sizeof(unsigned short int) * related_sequence_count);

        calc_diff_one_with_many(tile, related_tiles_for_tile, related_sequence_count, cache_info, diff_results);


        TilesSequence* new_sequence_first = (TilesSequence*)malloc(sizeof(TilesSequence));
        TilesSequence* new_sequence = new_sequence_first;
        TilesSequence* pre = NULL;

        unsigned int hit_count = 0;

        for (unsigned int i = 0; i < related_sequence_count; ++i) {
            if(diff_results[i] <= max_diff_pixels) {
                hit_count++;

                new_sequence->tile = related_tiles_for_tile[i];
                new_sequence->next = (TilesSequence*)malloc(sizeof(TilesSequence));
                pre = new_sequence;
                new_sequence = new_sequence->next;
                printf("pass %u - %u (%u)\n", tile->tile_id, related_tiles_for_tile[i]->tile_id, diff_results[i]);
                fflush(stdout);
            }
        }

        if(hit_count == 0) {
            free(new_sequence_first);
            delete_tiles_sequence(*related_sequence_for_tile);
            (*related_sequence_for_tile) = NULL;
        } else if(hit_count < related_sequence_count)  {
            free(new_sequence);
            pre->next = NULL;
            delete_tiles_sequence(*related_sequence_for_tile);
            (*related_sequence_for_tile) = new_sequence_first;
        } else { // ==
            new_sequence->next = NULL;
            delete_tiles_sequence(new_sequence_first);
        }

        free(related_tiles_for_tile);
        free(diff_results);
    } else if(related_sequence_count == 1) {
        const unsigned short t_compare_result = calc_diff(tile, (*related_sequence_for_tile)->tile, cache_info);

        if(t_compare_result > max_diff_pixels) {
            delete_tiles_sequence(*related_sequence_for_tile);
            (*related_sequence_for_tile) = NULL;
        } else {
            printf("pass %u - %u (%u)\n", tile->tile_id, (*related_sequence_for_tile)->tile->tile_id, t_compare_result);
            fflush(stdout);
        }
    }
}

NodeResult calc_node_result(TileGroupsSequence* groups_sequence,
                            const unsigned int groups_count,
                            BindedTilesSequence* binded_tiles_sequence,
                            GenericNode **used_tiles,
                            GenericNode **not_used,
                            const unsigned int unused_tiles_count,
                            Tile* const tile,
                            GenericNode* const related_tiles,
                            const unsigned int max_diff_pixels,
                            CacheInfo* const cache_info) {
    if(*not_used == NULL) {
        NodeResult result;
        result.created_groups_count = groups_count;
        result.rest_tiles_count = unused_tiles_count;
        return result;
    }

    migrate_tile(not_used, used_tiles, tile->tile_id);

    TilesSequence* related_tiles_sequence = (TilesSequence*) find(related_tiles, tile->tile_id)->data;

    ResultSlot left_slot;

    const NodeResult left_node_result = calc_left_node_result(groups_sequence,
                                                              groups_count,
                                                              used_tiles,
                                                              not_used,
                                                              binded_tiles_sequence,
                                                              unused_tiles_count,
                                                              tile,
                                                              related_tiles,
                                                              related_tiles_sequence,
                                                              max_diff_pixels,
                                                              cache_info);

    left_slot.tg_seq = groups_sequence->next;
    groups_sequence->next = NULL;

    left_slot.bt_seq = binded_tiles_sequence->next;
    binded_tiles_sequence->next = NULL;

    ResultSlot right_slot;
    groups_sequence->next = (TileGroupsSequence*)malloc(sizeof(TileGroupsSequence));
    groups_sequence->next->first = groups_sequence->first;
    groups_sequence->next->item = (TileGroup*)malloc(sizeof(TileGroup));
    groups_sequence->next->item->group_id = GROUP_ID_NOT_DEFINED;
    groups_sequence->next->item->leader_tile = tile;
    groups_sequence->next->next = NULL;

    const NodeResult right_node_result = calc_node_result_group_selected(groups_sequence->next,
                                                                         groups_count + 1,
                                                                         binded_tiles_sequence,
                                                                         used_tiles,
                                                                         not_used,
                                                                         unused_tiles_count - 1,
                                                                         related_tiles,
                                                                         related_tiles_sequence,
                                                                         max_diff_pixels,
                                                                         cache_info);

    right_slot.bt_seq = binded_tiles_sequence->next;
    binded_tiles_sequence->next = NULL;

    right_slot.tg_seq = groups_sequence->next;
    groups_sequence->next = NULL;

    const char best_result = choose_best_result(left_node_result, right_node_result);

    NodeResult best_node_result;

    if(best_result == LEFT_BEST) {
//        printf("g:%u ut:%u better than g:%u ut:%u (nd)\n",
//               left_node_result.created_groups_count,
//               left_node_result.rest_tiles_count,
//               right_node_result.created_groups_count,
//               right_node_result.rest_tiles_count);
//        fflush(stdout);
        delete_result_slot(right_slot);

        binded_tiles_sequence->next = left_slot.bt_seq;
        groups_sequence->next = left_slot.tg_seq;

        best_node_result = left_node_result;
    } else if(best_result == RIGHT_BEST || best_result == EQUAL) {
//        printf("g:%u ut:%u better than g:%u ut:%u (nd)\n",
//               right_node_result.created_groups_count,
//               right_node_result.rest_tiles_count,
//               left_node_result.created_groups_count,
//               left_node_result.rest_tiles_count);
//        fflush(stdout);
        delete_result_slot(left_slot);

        binded_tiles_sequence->next = right_slot.bt_seq;
        groups_sequence->next = right_slot.tg_seq;

        best_node_result = right_node_result;
    }

    migrate_tile(used_tiles, not_used, tile->tile_id);

    return best_node_result;
}


NodeResult calc_left_node_result(TileGroupsSequence* groups_sequence,
                                 const unsigned int groups_count,
                                 GenericNode **used_tiles,
                                 GenericNode **not_used,
                                 BindedTilesSequence *binded_tiles_sequence,
                                 const unsigned int unused_tiles_count,
                                 Tile *const tile,
                                 GenericNode* const related_tiles,
                                 TilesSequence * related_tiles_sequence,
                                 const unsigned int max_diff_pixels,
                                 CacheInfo* const cache_info) {
    ResultSlot used_slot;
    used_slot.bt_seq = NULL;
    used_slot.tg_seq = NULL;
    used_slot.is_empty = 1;

    if(groups_sequence == NULL) { // no groups
    } else {
        TileGroupsSequence* t_group_sequence = groups_sequence->first;

        unsigned short int t_diff_result = USHORT_MAX;

        ResultSlot empty_slot;
        empty_slot.bt_seq = NULL;
        empty_slot.tg_seq = NULL;
        empty_slot.is_empty = 1;

        ResultSlot t_slot;

        char t_best_result = EQUAL;

        while (t_group_sequence != NULL) {
            t_diff_result = calc_diff(tile, t_group_sequence->item->leader_tile, cache_info);

            if(t_diff_result <= max_diff_pixels) {
                binded_tiles_sequence->next = (BindedTilesSequence*)malloc(sizeof(BindedTilesSequence));
                binded_tiles_sequence->next->item = (BindedTile*)malloc(sizeof(BindedTile));
                binded_tiles_sequence->next->item->group = t_group_sequence->item;
                binded_tiles_sequence->next->item->tile = tile;
                binded_tiles_sequence->next->next = NULL;


                empty_slot.node_result = calc_node_result_group_selected(groups_sequence,
                                                                         groups_count,
                                                                         binded_tiles_sequence->next,
                                                                         used_tiles,
                                                                         not_used,
                                                                         unused_tiles_count - 1,
                                                                         related_tiles,
                                                                         related_tiles_sequence,
                                                                         max_diff_pixels,
                                                                         cache_info);


                empty_slot.tg_seq = groups_sequence->next;
                groups_sequence->next = NULL;

                empty_slot.bt_seq = binded_tiles_sequence->next;
                binded_tiles_sequence->next = NULL;

                empty_slot.is_empty = 0;

                if(!used_slot.is_empty) {
                    t_best_result = choose_best_result(empty_slot.node_result, used_slot.node_result);

                    if(t_best_result == LEFT_BEST) {
//                        printf("g:%u ut:%u better than g:%u ut:%u (lfn)\n",
//                               empty_slot.node_result.created_groups_count,
//                               empty_slot.node_result.rest_tiles_count,
//                               used_slot.node_result.created_groups_count,
//                               used_slot.node_result.rest_tiles_count);
                        delete_result_slot(used_slot);

                        t_slot = used_slot;
                        used_slot = empty_slot;
                        empty_slot = t_slot;
                    } else if(t_best_result == RIGHT_BEST || t_best_result == EQUAL) {
//                        printf("g:%u ut:%u better than g:%u ut:%u (lfn)\n",
//                               used_slot.node_result.created_groups_count,
//                               used_slot.node_result.rest_tiles_count,
//                               empty_slot.node_result.created_groups_count,
//                               empty_slot.node_result.rest_tiles_count);
                        delete_result_slot(empty_slot);
                    }
                } else {
                    t_slot = used_slot;
                    used_slot = empty_slot;
                    empty_slot = t_slot;
                }
            }

            t_group_sequence = t_group_sequence->next;
        }
    }

    if(!used_slot.is_empty) {
        groups_sequence->next = used_slot.tg_seq;
        binded_tiles_sequence->next = used_slot.bt_seq;

        return used_slot.node_result;
    } else {
        if(*not_used != NULL) {
            return calc_node_result(groups_sequence,
                                    groups_count,
                                    binded_tiles_sequence,
                                    used_tiles,
                                    not_used,
                                    unused_tiles_count,
                                    (Tile*)(*not_used)->data,
                                    related_tiles,
                                    max_diff_pixels,
                                    cache_info);
        } else {
            NodeResult result;
            result.created_groups_count = groups_count;
            result.rest_tiles_count = unused_tiles_count;
            return result;
        }
    }
}


NodeResult calc_node_result_group_selected(TileGroupsSequence* groups_sequence,
                                           const unsigned int groups_count,
                                           BindedTilesSequence* binded_tiles_sequence,
                                           GenericNode** used_tiles,
                                           GenericNode** not_used,
                                           const unsigned int unused_tiles_count,
                                           GenericNode* const related_tiles,
                                           TilesSequence* related_tiles_sequence,
                                           const unsigned int max_diff_pixels,
                                           CacheInfo* const cache_info) {
    if(*not_used == NULL) {
        NodeResult result;
        result.created_groups_count = groups_count;
        result.rest_tiles_count = unused_tiles_count;
        return result;
    }

    ResultSlot used_slot;
    used_slot.bt_seq = NULL;
    used_slot.tg_seq = NULL;
    used_slot.is_empty = 1;

    if(related_tiles_sequence == NULL) { // no related
    } else {
        TilesSequence* t_related_tile_sequence = related_tiles_sequence;

        ResultSlot empty_slot;
        empty_slot.bt_seq = NULL;
        empty_slot.tg_seq = NULL;
        empty_slot.is_empty = 1;


        ResultSlot t_slot;

        char t_best_result = EQUAL;

        while(t_related_tile_sequence != NULL) {
            if (find(*used_tiles, t_related_tile_sequence->tile->tile_id) != NULL) { // tile already used
            } else {
                empty_slot.node_result = calc_node_result(groups_sequence,
                                                          groups_count,
                                                          binded_tiles_sequence,
                                                          used_tiles,
                                                          not_used,
                                                          unused_tiles_count,
                                                          t_related_tile_sequence->tile,
                                                          related_tiles,
                                                          max_diff_pixels,
                                                          cache_info);

                empty_slot.tg_seq = groups_sequence->next;
                groups_sequence->next = NULL;

                empty_slot.bt_seq = binded_tiles_sequence->next;
                binded_tiles_sequence->next = NULL;

                empty_slot.is_empty = 0;

                if(!used_slot.is_empty) {
                    t_best_result = choose_best_result(empty_slot.node_result, used_slot.node_result);

                    if(t_best_result == LEFT_BEST) {
//                        printf("g:%u ut:%u better than g:%u ut:%u (lfngs)\n",
//                               empty_slot.node_result.created_groups_count,
//                               empty_slot.node_result.rest_tiles_count,
//                               used_slot.node_result.created_groups_count,
//                               used_slot.node_result.rest_tiles_count);
                        delete_result_slot(used_slot);

                        t_slot = used_slot;
                        used_slot = empty_slot;
                        empty_slot = t_slot;
                    } else if(t_best_result == RIGHT_BEST || t_best_result == EQUAL) {
//                        printf("g:%u ut:%u better than g:%u ut:%u(lfngs)\n",
//                               used_slot.node_result.created_groups_count,
//                               used_slot.node_result.rest_tiles_count,
//                               empty_slot.node_result.created_groups_count,
//                               empty_slot.node_result.rest_tiles_count);
                        delete_result_slot(empty_slot);
                    }
                } else {
                    t_slot = used_slot;
                    used_slot = empty_slot;
                    empty_slot = t_slot;
                }
            }

            t_related_tile_sequence = t_related_tile_sequence->next;
        }
    }


    if(used_slot.is_empty) {
        return calc_node_result(groups_sequence,
                                groups_count,
                                binded_tiles_sequence,
                                used_tiles,
                                not_used,
                                unused_tiles_count,
                                (Tile*)(*not_used)->data,
                                related_tiles,
                                max_diff_pixels,
                                cache_info);
    } else {
        groups_sequence->next = used_slot.tg_seq;
        binded_tiles_sequence->next = used_slot.bt_seq;

        return used_slot.node_result;
    }
}


void hungry_by_groups(TileGroupsSequence* groups_sequence,
                            BindedTilesSequence* binded_tiles_sequence,
                            GenericNode** used_tiles,
                            GenericNode** not_used,
                            Tile *const tile,
                            const unsigned int max_diff_pixels,
                              CacheInfo* const cache_info) {

    migrate_tile(not_used, used_tiles, tile->tile_id);

    ResultSlot left_slot;

    const unsigned char left_node_result = hungry_by_groups_left(groups_sequence,
                                                              binded_tiles_sequence,
                                                              tile,
                                                              max_diff_pixels,
                                                              cache_info);
    if(left_node_result) {
    } else {
        left_slot.tg_seq = groups_sequence->next;
        groups_sequence->next = NULL;

        left_slot.bt_seq = binded_tiles_sequence->next;
        binded_tiles_sequence->next = NULL;

        delete_result_slot(left_slot);

        groups_sequence->next = (TileGroupsSequence*)malloc(sizeof(TileGroupsSequence));
        groups_sequence->next->first = groups_sequence->first;
        groups_sequence->next->item = tile;
        groups_sequence->next->next = NULL;
    }

    if(*not_used != NULL) {
        hungry_by_groups(groups_sequence, binded_tiles_sequence, used_tiles, not_used, (Tile*)(*not_used)->data, max_diff_pixels, cache_info);
    }
}

unsigned char hungry_by_groups_left(TileGroupsSequence* groups_sequence,
                                 BindedTilesSequence* binded_tiles_sequence,
                                 Tile* const tile,
                                 const unsigned int max_diff_pixels,
                                 CacheInfo* const cache_info) {
    if(groups_sequence == NULL) { // no groups
    } else {
        TileGroupsSequence* t_group_sequence = groups_sequence->first;

        unsigned short int t_diff_result = USHORT_MAX;

        while (t_group_sequence != NULL) {
            t_diff_result = calc_diff(tile, t_group_sequence->item->leader_tile, cache_info);

            if(t_diff_result <= max_diff_pixels) {
                binded_tiles_sequence->next = (BindedTilesSequence*)malloc(sizeof(BindedTilesSequence));
                binded_tiles_sequence->next->item = (BindedTile*)malloc(sizeof(BindedTile));
                binded_tiles_sequence->next->item->group = t_group_sequence->item;
                binded_tiles_sequence->next->item->tile = tile;
                binded_tiles_sequence->next->next = NULL;

                return 1;
            }

            t_group_sequence = t_group_sequence->next;
        }
    }

    return 0;
}

TilesSequence* make_tile_sequence_from_tree(const GenericNode* const node, TilesSequence* const sequence) {
    if(node == NULL) {
        sequence->tile = NULL;
        sequence->next = NULL;
        return sequence;
    }

    TilesSequence* const last_elem = make_tile_sequence_from_tree(node->left, sequence);

    last_elem->tile = (Tile*)node->data;
    last_elem->next = (TilesSequence*)malloc(sizeof(TilesSequence));

    return make_tile_sequence_from_tree(node->right, last_elem->next);
}

void delete_tiles_sequence(TilesSequence* const tile_sequence) {
    if(tile_sequence == NULL)
        return;

    delete_tiles_sequence(tile_sequence->next);

    free(tile_sequence);
}

void tile_sequence_destructor(void* data) {
    delete_tiles_sequence((TilesSequence*)data);
}

GenericNode* create_tiles_tree_from_tiles_ids(GenericNode* const all_tiles,
                                              const unsigned int* const ids,
                                              const unsigned int count) {
    if(count < 1) {
        return NULL;
    }

    GenericNode* root = create_node(ids[0], find(all_tiles, ids[0])->data);

    for (unsigned int i = 1; i < count; ++i) {
        root = insert(root, ids[i], find(all_tiles, ids[i])->data);
    }

    return root;
}

TilesSequence* create_tiles_sequence_from_tile_ids(GenericNode* const tiles,
                                                   const unsigned int* const ids,
                                                   const unsigned int count) {

    if(count < 1) {
        return NULL;
    }

    TilesSequence* const first = (TilesSequence*)malloc(sizeof(TilesSequence));

    first->tile = (Tile*) find(tiles, ids[0])->data;

    if(count == 1) {
        first->next = NULL;
        return first;
    }

    TilesSequence* t_next = (TilesSequence*)malloc(sizeof(TilesSequence));
    t_next->tile = (Tile*) find(tiles, ids[1])->data;

    first->next = t_next;

    for (unsigned int i = 2; i < count; ++i) {
        t_next->next = (TilesSequence*)malloc(sizeof(TilesSequence));
        t_next = t_next->next;
        t_next->tile = (Tile*)find(tiles, ids[i])->data;
    }

    t_next->next = NULL;

    return first;
}

TileGroupsSequence* create_tiles_groups_sequence_from_ids(GenericNode* const all_tiles,
                                                          const unsigned int* const groups_ids,
                                                          const unsigned int* const leader_tiles_ids,
                                                          const unsigned int count) {
    if(count < 1) {
        return NULL;
    }

    TileGroupsSequence* const first = (TileGroupsSequence*)malloc(sizeof(TileGroupsSequence));
    first->first = first;

    first->item = (TileGroup*)malloc(sizeof(TileGroup));
    first->item->group_id = groups_ids[0];
    first->item->leader_tile = (Tile*) find(all_tiles, leader_tiles_ids[0])->data;

    if(count == 1) {
        first->next = NULL;
        return first;
    }

    TileGroupsSequence* t_next = (TileGroupsSequence*)malloc(sizeof(TileGroupsSequence));
    t_next->first = first;
    t_next->item = (TileGroup*)malloc(sizeof(TileGroup));;
    t_next->item->group_id = groups_ids[1];
    t_next->item->leader_tile = (Tile*) find(all_tiles, leader_tiles_ids[1])->data;

    first->next = t_next;

    for (unsigned int i = 2; i < count; ++i) {
        t_next->next = (TileGroupsSequence*)malloc(sizeof(TileGroupsSequence));
        t_next = t_next->next;

        t_next->item->group_id = groups_ids[1];
        t_next->item->leader_tile = (Tile*) find(all_tiles, leader_tiles_ids[1])->data;
        t_next->first = first;
    }

    t_next->next = NULL;

    return first;
}

char choose_best_result(const NodeResult left_node_result, const NodeResult right_node_result) {
    const unsigned int l_rating = left_node_result.created_groups_count + left_node_result.rest_tiles_count;
    const unsigned int r_rating = right_node_result.created_groups_count + right_node_result.rest_tiles_count;

    return (LEFT_BEST) * (l_rating < r_rating) + (RIGHT_BEST) * (l_rating > r_rating); // -1 | 0 | 1
}
