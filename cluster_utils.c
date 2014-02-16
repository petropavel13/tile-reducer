#include "cluster_utils.h"

void make_persistent_groups(const DbInfo *const db_info,
                            GenericNode *const tiles_root_node,
                            const unsigned int total,
                            const AppRunParams arp,
                            CacheInfo* const cache_info,
                            void (*callback)(unsigned int, unsigned int)) {
    unsigned int* pg_ids = NULL;
    unsigned int t_ids_count = 0;

    const Tile* next_tile = NULL;

    const Tile** const equal_candidates = (const Tile**)malloc(sizeof(Tile*) * total); // prevent many malloc/free calls

    unsigned short int* const results = malloc(sizeof(unsigned short int) * total); // prevent many malloc/free calls

    unsigned int processed_count = 0;

    unsigned int passed_ids_count = 0;

    const GenericNode* t_next_node = find(tiles_root_node, load_next_zero_equal_id_leader(db_info));

    while(t_next_node != NULL) {
        next_tile = t_next_node->data;

        load_zero_equals_ids_for_tile(db_info, next_tile->tile_id, &pg_ids, &t_ids_count);

        if(callback != NULL) {
            callback(processed_count, t_ids_count);
        }

        for (unsigned int i = 0; i < t_ids_count; ++i) {
            equal_candidates[i] = (Tile*) find(tiles_root_node, pg_ids[i])->data;
        }

        if(t_ids_count > 1) {
            calc_diff_one_with_many(next_tile, equal_candidates, t_ids_count, cache_info, arp, results);

            for (unsigned int i = 0; i < t_ids_count; ++i) {
                if(results[i] == 0) {
                    add_tile_to_persistent_group_using_buffer(db_info, next_tile->tile_id, equal_candidates[i]->tile_id);
                    pg_ids[passed_ids_count++] = equal_candidates[i]->tile_id; // reuse pg_ids
                }
            }

            flush_db_buffer(db_info); // add to persistent_groups

            for (unsigned int i = 0; i < passed_ids_count; ++i) {
                delete_zero_equal_pair_using_buffer(db_info, next_tile->tile_id, pg_ids[i]);
            }

            flush_db_buffer(db_info); // delete from count_equality_mv

            if(passed_ids_count != t_ids_count) {
                remix_zero_equals_ids(db_info, next_tile->tile_id);
            }

            passed_ids_count = 0;
        } else if(t_ids_count == 1) {  // compare 1 to 1 (use simpler functions)
            delete_zero_equal_pair(db_info, next_tile->tile_id, equal_candidates[0]->tile_id);

            if(calc_diff(next_tile, equal_candidates[0], cache_info) == 0) {
                add_tile_to_persistent_group(db_info, equal_candidates[0]->tile_id, next_tile->tile_id);
            }
        }

        free(pg_ids); pg_ids = NULL;

        processed_count++;

        t_ids_count = 0;

        t_next_node = find(tiles_root_node, load_next_zero_equal_id_leader(db_info));
    }

    free(equal_candidates);
    free(results);
}

void migrate_tile(GenericNode** from, GenericNode** to, const unsigned int tile_id) {
    *to = insert(*to, tile_id, find(*from, tile_id)->data);
    *from = remove_node(*from, tile_id, NULL);
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


void clusterize_simple(GenericNode *const all_tiles,
                const unsigned int all_tiles_count,
                const AppRunParams arp,
                const DbInfo* const db_info,
                CacheInfo* const cache_info) {
    unsigned int* const not_persistent_ids = malloc(sizeof(unsigned int) * all_tiles_count);

    clear_working_set(db_info);
    create_working_set_wo_persistent_ids(db_info);

    unsigned int not_persistent_count = 0;

    read_working_set_ids(db_info, not_persistent_ids, &not_persistent_count);

    Tile** const not_persistent_tiles = malloc(sizeof(Tile*) * not_persistent_count);

    for (unsigned int i = 0; i < not_persistent_count; ++i) {
        not_persistent_tiles[i] = find(all_tiles, not_persistent_ids[i])->data;
    }

    free(not_persistent_ids);

    GenericNode* cached_relaterd_results = NULL;

    unsigned int* t_related_ids = malloc(sizeof(unsigned int) * not_persistent_count);
    unsigned int t_related_count = 0;
    Tile** t_related_tiles = malloc(sizeof(Tile*) * not_persistent_count);

    Tile* t_current;

    GenericNode* free_tiles = NULL;

    for (unsigned int i = 0; i < not_persistent_count; ++i) {
        t_current = not_persistent_tiles[i];

        if (find(cached_relaterd_results, t_current->tile_id) != NULL) {
            free_tiles = insert(free_tiles, t_current->tile_id, t_current);
            continue;
        }

        read_related_tiles_ids(db_info, t_current->tile_id, t_related_ids, &t_related_count, arp.max_diff_pixels);

        if (t_related_count == 0) {
            continue;
        }

        for (unsigned int j = 0; j < t_related_count; ++j) {
            t_related_tiles[j] = find(all_tiles, t_related_ids[j])->data;
        }

        if (check_has_real_related(t_current, (const Tile* const * const)t_related_tiles, t_related_count, &cached_relaterd_results, arp, cache_info) == 1) {
            free_tiles = insert(free_tiles, t_current->tile_id, t_current);
        }
    }

    free(t_related_ids);
    free(t_related_tiles);

    destroy_tree(cached_relaterd_results, NULL);

    GenericNode* used_tiles = create_node(0, NULL); // dummy

    unsigned int* persistent_groups_leaders_ids = NULL;
    unsigned int persistent_groups_count = 0;

    read_persistent_groups(db_info, &persistent_groups_leaders_ids, &persistent_groups_count);

    Tile** const groups = malloc(sizeof(Tile*) * (not_persistent_count + persistent_groups_count));

    for (unsigned int i = 0; i < persistent_groups_count; ++i) {
        groups[i] = (Tile*) find(all_tiles, persistent_groups_leaders_ids[i])->data;
    }

    BindedTilesSequence* const binded_tiles_sequence = malloc(sizeof(BindedTilesSequence));
    binded_tiles_sequence->item = NULL;
    binded_tiles_sequence->next = NULL;


    simple_by_groups(groups, &persistent_groups_count, binded_tiles_sequence, &used_tiles, &free_tiles, (Tile*) free_tiles->data, arp.max_diff_pixels, cache_info);

    BindedTilesSequence* t_binded_sequence = binded_tiles_sequence->next;

    while (t_binded_sequence->next != NULL) {
        add_tile_to_group_using_buffer(db_info, t_binded_sequence->item->group_leader_tile->tile_id, t_binded_sequence->item->tile->tile_id);

        t_binded_sequence = t_binded_sequence->next;
    }

    flush_db_buffer(db_info);


    free(groups);
    delete_binded_tiles_sequence(binded_tiles_sequence);

    destroy_tree(free_tiles, NULL);
    destroy_tree(used_tiles, NULL);
}

unsigned char check_has_real_related(const Tile* const tile,
                                     const Tile* const * const related_tiles,
                                     const unsigned int related_count,
                                     GenericNode** const cached_related,
                                     const AppRunParams arp,
                                     CacheInfo* const cache_info) {
    if (related_count == 1) {
        unsigned char yes = calc_diff(tile, related_tiles[0], cache_info) <= arp.max_diff_pixels;

        if (yes) {
            *cached_related = insert(*cached_related, related_tiles[0]->tile_id, NULL);
        }

        return yes;
    }

    unsigned short* const diff_results = malloc(sizeof(unsigned short) * related_count);

    calc_diff_one_with_many(tile, related_tiles, related_count, cache_info, arp, diff_results);

    unsigned char yes = 0;

    for (unsigned int i = 0; i < related_count; ++i) {
        if (diff_results[i] <= arp.max_diff_pixels) {
            *cached_related = insert(*cached_related, related_tiles[i]->tile_id, NULL);
            yes = 1;
        }
    }

    free(diff_results);

    return yes;
}

void simple_by_groups(Tile** const groups,
                      unsigned int *offset,
                      BindedTilesSequence* binded_tiles_sequence,
                      GenericNode** used_tiles,
                      GenericNode** not_used,
                      Tile *const tile,
                      const unsigned int max_diff_pixels,
                      CacheInfo* const cache_info) {
    migrate_tile(not_used, used_tiles, tile->tile_id);

    const unsigned char left_node_result = simple_by_groups_left(groups,
                                                                 offset,
                                                                 binded_tiles_sequence,
                                                                 tile,
                                                                 max_diff_pixels,
                                                                 cache_info);
    if(left_node_result) {
        if(*not_used != NULL) {
            simple_by_groups(groups, offset, binded_tiles_sequence->next, used_tiles, not_used, (Tile*)(*not_used)->data, max_diff_pixels, cache_info);
        }
    } else {
        groups[(*offset)++] = tile;

        if(*not_used != NULL) {
            simple_by_groups(groups, offset, binded_tiles_sequence, used_tiles, not_used, (Tile*)(*not_used)->data, max_diff_pixels, cache_info);
        }
    }

}

unsigned char simple_by_groups_left(Tile** const groups,
                                    unsigned int* const offset,
                                    BindedTilesSequence* binded_tiles_sequence,
                                    Tile* const tile,
                                    const unsigned int max_diff_pixels,
                                    CacheInfo* const cache_info) {
    Tile* t_group_leader;

    unsigned short int t_diff_result = USHORT_MAX;

    // TODO compare one with others!

    for (unsigned int i = 0; i < (*offset); ++i) {
        t_group_leader = groups[i];

        t_diff_result = calc_diff(tile, t_group_leader, cache_info);

        if(t_diff_result <= max_diff_pixels) {
            binded_tiles_sequence->next = malloc(sizeof(BindedTilesSequence));
            binded_tiles_sequence->next->item = malloc(sizeof(BindedTile));
            binded_tiles_sequence->next->item->group_leader_tile = t_group_leader;
            binded_tiles_sequence->next->item->tile = tile;
            binded_tiles_sequence->next->next = NULL;

            return 1;
        }
    }

    return 0;
}

//void clusterize(GenericNode* const all_tiles,
//                const unsigned int all_tiles_count,
//                const AppRunParams arp,
//                const DbInfo* const db_info,
//                CacheInfo* const cache_info) {
//    unsigned int* const possible_leaders_ids = malloc(sizeof(unsigned int) * all_tiles_count);

//    materialize_tile_color_count_wo_persistent(db_info);

//    unsigned int possible_leaders_count = 0;

//    read_tile_color_count_tiles_ids(db_info, possible_leaders_ids, &possible_leaders_count);

//    GenericNode* free_tiles = create_tiles_tree_from_tiles_ids(all_tiles, possible_leaders_ids, possible_leaders_count);

//    free(possible_leaders_ids);

//    GenericNode* used_tiles = create_node(0, NULL); // dummy

//    unsigned int* persistent_groups_leaders_ids = NULL;
//    unsigned int persistent_groups_count = 0;

//    read_persistent_groups(db_info, &persistent_groups_leaders_ids, &persistent_groups_count);

//    TileGroupsSequence* const tile_group_sequence = create_tiles_groups_sequence_from_ids(all_tiles, persistent_groups_leaders_ids, persistent_groups_count);

//    BindedTilesSequence* const binded_tiles_sequence = (BindedTilesSequence*)malloc(sizeof(BindedTilesSequence));
//    binded_tiles_sequence->item = NULL;
//    binded_tiles_sequence->next = NULL;

//    TilesSequence* t_tiles_sequence_from_free = (TilesSequence*)malloc(sizeof(TilesSequence));
//    t_tiles_sequence_from_free->tile = NULL;
//    t_tiles_sequence_from_free->next = NULL;

//    TilesSequence* t_last = make_tile_sequence_from_tree(free_tiles, t_tiles_sequence_from_free);

//    TilesSequence* t_tiles_sequence = t_tiles_sequence_from_free;

//    GenericNode* related_tiles = NULL;

//    unsigned int* const related_ids = malloc(sizeof(unsigned int) * possible_leaders_count);

//    unsigned int related_count = 0;

//    TilesSequence* t_related_sequence = NULL;

//    while (t_tiles_sequence != NULL && t_tiles_sequence != t_last) {
//        read_related_tiles_ids(db_info, t_tiles_sequence->tile->tile_id, related_ids, &related_count, arp.max_diff_pixels);

//        t_related_sequence = create_tiles_sequence_from_tile_ids(free_tiles, related_ids, related_count);

//        if(t_related_sequence == NULL) {
//            possible_leaders_count--;
//            free_tiles = remove_node(free_tiles, t_tiles_sequence->tile->tile_id, NULL);
//        } else {
//            clean_related_group(t_tiles_sequence->tile, &t_related_sequence, related_count, arp, cache_info);
//            related_tiles = insert(related_tiles, t_tiles_sequence->tile->tile_id, t_related_sequence);
//        }

//        t_tiles_sequence = t_tiles_sequence->next;
//    }

//    delete_tiles_sequence(t_tiles_sequence_from_free);

//    t_tiles_sequence_from_free = (TilesSequence*)malloc(sizeof(TilesSequence));
//    t_tiles_sequence_from_free->tile = NULL;
//    t_tiles_sequence_from_free->next = NULL;

//    t_last = make_tile_sequence_from_tree(free_tiles, t_tiles_sequence_from_free);

//    t_tiles_sequence = t_tiles_sequence_from_free;

//    while (t_tiles_sequence != NULL && t_tiles_sequence != t_last) {
//        t_related_sequence = (TilesSequence*) find(related_tiles, t_tiles_sequence->tile->tile_id)->data;

//        if(t_related_sequence == NULL) {
//            possible_leaders_count--;
//            free_tiles = remove_node(free_tiles, t_tiles_sequence->tile->tile_id, NULL);
//            related_tiles = remove_node(related_tiles, t_tiles_sequence->tile->tile_id, &tile_sequence_destructor);
//        }

//        t_tiles_sequence = t_tiles_sequence->next;
//    }

////    unsigned long f_count = 0;
////    calc_elements_count(free_tiles, &f_count);

//    free(related_ids);

////    const NodeResult best_result = calc_node_result(tile_group_sequence,
////                                                    persistent_groups_count,
////                                                    binded_tiles_sequence,
////                                                    &used_tiles,
////                                                    &free_tiles,
////                                                    working_set_count,
////                                                    (Tile*) free_tiles->data,
////                                                    related_tiles,
////                                                    max_diff_pixels,
////                                                    cache_info);

//    TileGroupsSequence* last_group_in_seq = tile_group_sequence;

//    while (last_group_in_seq->next != NULL) {
//        last_group_in_seq = last_group_in_seq->next;
//    }

//    simple_by_groups(last_group_in_seq, binded_tiles_sequence, &used_tiles, &free_tiles, (Tile*) free_tiles->data, arp.max_diff_pixels, cache_info);

//    BindedTilesSequence* t_binded_sequence = binded_tiles_sequence->next;

//    while (t_binded_sequence->next != NULL) {
//        add_tile_to_group_using_buffer(db_info, t_binded_sequence->item->group_leader_tile->tile_id, t_binded_sequence->item->tile->tile_id);

//        t_binded_sequence = t_binded_sequence->next;
//    }

//    flush_db_buffer(db_info);


//    delete_tile_groups_sequence(tile_group_sequence);
//    delete_binded_tiles_sequence(binded_tiles_sequence);

//    destroy_tree(free_tiles, NULL);
//    destroy_tree(used_tiles, NULL);
//    destroy_tree(related_tiles, &tile_sequence_destructor);


//    delete_tiles_sequence(t_tiles_sequence_from_free);
//}



void clean_related_group(const Tile* const tile,
                         TilesSequence** related_sequence_for_tile,
                         const unsigned int related_sequence_count,
                         const AppRunParams arp,
                         CacheInfo* const cache_info) {
    if(related_sequence_count > 1) {
        Tile* related_tiles_for_tile[related_sequence_count];

        TilesSequence* t_related_tile = *related_sequence_for_tile;

        for (unsigned int i = 0; i < related_sequence_count; ++i) {
            related_tiles_for_tile[i] = t_related_tile->tile;
            t_related_tile = t_related_tile->next;
        }

        unsigned short int* const diff_results = (unsigned short int*)malloc(sizeof(unsigned short int) * related_sequence_count);

        calc_diff_one_with_many(tile, (const Tile* const * const)related_tiles_for_tile, related_sequence_count, cache_info, arp, diff_results);


        TilesSequence* new_sequence_first = (TilesSequence*)malloc(sizeof(TilesSequence));
        TilesSequence* new_sequence = new_sequence_first;
        TilesSequence* pre = NULL;

        unsigned int hit_count = 0;

        for (unsigned int i = 0; i < related_sequence_count; ++i) {
            if(diff_results[i] <= arp.max_diff_pixels) {
                hit_count++;

                new_sequence->tile = related_tiles_for_tile[i];
                new_sequence->next = (TilesSequence*)malloc(sizeof(TilesSequence));
                pre = new_sequence;
                new_sequence = new_sequence->next;
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

        free(diff_results);
    } else if(related_sequence_count == 1) {
        const unsigned short t_compare_result = calc_diff(tile, (*related_sequence_for_tile)->tile, cache_info);

        if(t_compare_result > arp.max_diff_pixels) {
            delete_tiles_sequence(*related_sequence_for_tile);
            (*related_sequence_for_tile) = NULL;
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
    groups_sequence->next->leader_tile = tile;
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
        delete_result_slot(right_slot);

        binded_tiles_sequence->next = left_slot.bt_seq;
        groups_sequence->next = left_slot.tg_seq;

        best_node_result = left_node_result;
    } else if(best_result == RIGHT_BEST || best_result == EQUAL) {
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
            t_diff_result = calc_diff(tile, t_group_sequence->leader_tile, cache_info);

            if(t_diff_result <= max_diff_pixels) {
                binded_tiles_sequence->next = (BindedTilesSequence*)malloc(sizeof(BindedTilesSequence));
                binded_tiles_sequence->next->item = (BindedTile*)malloc(sizeof(BindedTile));
                binded_tiles_sequence->next->item->group_leader_tile = t_group_sequence->leader_tile;
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
                        delete_result_slot(used_slot);

                        t_slot = used_slot;
                        used_slot = empty_slot;
                        empty_slot = t_slot;
                    } else if(t_best_result == RIGHT_BEST || t_best_result == EQUAL) {
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
                        delete_result_slot(used_slot);

                        t_slot = used_slot;
                        used_slot = empty_slot;
                        empty_slot = t_slot;
                    } else if(t_best_result == RIGHT_BEST || t_best_result == EQUAL) {
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
                                                          const unsigned int* const leader_tiles_ids,
                                                          const unsigned int count) {
    if(count < 1) {
        return NULL;
    }

    TileGroupsSequence* const first = (TileGroupsSequence*)malloc(sizeof(TileGroupsSequence));
    first->first = first;

    first->leader_tile = (Tile*) find(all_tiles, leader_tiles_ids[0])->data;

    if(count == 1) {
        first->next = NULL;
        return first;
    }

    TileGroupsSequence* t_next = (TileGroupsSequence*)malloc(sizeof(TileGroupsSequence));
    t_next->first = first;
    t_next->leader_tile = (Tile*) find(all_tiles, leader_tiles_ids[1])->data;

    first->next = t_next;

    for (unsigned int i = 2; i < count; ++i) {
        t_next->next = (TileGroupsSequence*)malloc(sizeof(TileGroupsSequence));
        t_next = t_next->next;

        t_next->leader_tile = (Tile*) find(all_tiles, leader_tiles_ids[i])->data;
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
