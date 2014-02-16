#ifndef CLUSTER_UTILS_H
#define CLUSTER_UTILS_H

#include "tile_utils.h"
#include "db_utils.h"
#include "apprunparams.h"

#include <stdio.h>

typedef struct TilesSequence {
    Tile* tile;
    struct TilesSequence* next;
} TilesSequence;


typedef struct TileGroupsSequence {
    struct TileGroupsSequence* first;
    Tile* leader_tile;
    struct TileGroupsSequence* next;
} TileGroupsSequence;


typedef struct BindedTile {
    Tile* tile;
    Tile* group_leader_tile;
} BindedTile;

typedef struct BindedTilesSequence {
    BindedTile* item;
    struct BindedTilesSequence* next;
} BindedTilesSequence;


typedef struct NodeResult {
    unsigned int created_groups_count;
    unsigned int rest_tiles_count;
} NodeResult;


typedef struct ResultSlot {
    BindedTilesSequence* bt_seq;
    TileGroupsSequence* tg_seq;
    NodeResult node_result;
    unsigned char is_empty;
} ResultSlot;



#define GROUP_ID_NOT_DEFINED 0

#define LEFT_BEST (-1)
#define EQUAL 0
#define RIGHT_BEST 1


void make_persistent_groups(const DbInfo* const db_info,
                            GenericNode *const tiles_root_node,
                            const unsigned int total,
                            const AppRunParams arp,
                            CacheInfo *const cache_info,
                            void (*callback)(unsigned int, unsigned int));


void migrate_tile(GenericNode** from, GenericNode** to, const unsigned int tile_id);

void delete_tile_groups_sequence(TileGroupsSequence* const sequence);

void delete_binded_tiles_sequence(BindedTilesSequence* const sequence);

void delete_result_slot(ResultSlot slot);

void clusterize_simple(GenericNode *const all_tiles,
                const unsigned int all_tiles_count,
                const AppRunParams arp,
                const DbInfo* const db_info,
                CacheInfo* const cache_info);

unsigned char check_has_real_related(const Tile* const tile,
                                     const Tile* const * const related_tiles,
                                     const unsigned int related_count,
                                     GenericNode ** const cached_related,
                                     const AppRunParams arp,
                                     CacheInfo* const cache_info);

void simple_by_groups(Tile ** const groups,
                      unsigned int* offset,
                      BindedTilesSequence* binded_tiles_sequence,
                      GenericNode** used_tiles,
                      GenericNode** not_used,
                      Tile *const tile,
                      const unsigned int max_diff_pixels,
                      CacheInfo* const cache_info);


unsigned char simple_by_groups_left(Tile ** const groups,
                                    unsigned int * const offset,
                                    BindedTilesSequence* binded_tiles_sequence,
                                    Tile* const tile,
                                    const unsigned int max_diff_pixels,
                                    CacheInfo* const cache_info);

//void clusterize(GenericNode *const all_tiles,
//                const unsigned int all_tiles_count,
//                const AppRunParams arp,
//                const DbInfo* const db_info,
//                CacheInfo* const cache_info);



void clean_related_group(const Tile *const tile,
                         TilesSequence **related_sequence_for_tile,
                         const unsigned int related_sequence_count,
                         const AppRunParams arp,
                         CacheInfo* const cache_info);

NodeResult calc_node_result(TileGroupsSequence* groups_sequence,
                            const unsigned int groups_count,
                            BindedTilesSequence* binded_tiles_sequence,
                            GenericNode** used_tiles,
                            GenericNode** not_used,
                            const unsigned int unused_tiles_count,
                            Tile *const tile,
                            GenericNode *const related_tiles,
                            const unsigned int max_diff_pixels,
                            CacheInfo* const cache_info);


NodeResult calc_left_node_result(TileGroupsSequence* groups_sequence,
                                 const unsigned int groups_count,
                                 GenericNode** used_tiles,
                                 GenericNode** not_used,
                                 BindedTilesSequence* binded_tiles_sequence,
                                 const unsigned int unused_tiles_count,
                                 Tile* const tile,
                                 GenericNode *const related_tiles,
                                 TilesSequence *related_tiles_sequence,
                                 const unsigned int max_diff_pixels,
                                 CacheInfo* const cache_info);


NodeResult calc_node_result_group_selected(TileGroupsSequence* groups_sequence,
                                           const unsigned int groups_count,
                                           BindedTilesSequence* binded_tiles_sequence,
                                           GenericNode** used_tiles,
                                           GenericNode** not_used,
                                           const unsigned int unused_tiles_count,
                                           GenericNode *const related_tiles,
                                           TilesSequence *related_tiles_sequence,
                                           const unsigned int max_diff_pixels,
                                           CacheInfo* const cache_info);




TilesSequence *make_tile_sequence_from_tree(const GenericNode* const node, TilesSequence* const sequence);

void delete_tiles_sequence(TilesSequence* const tile_sequence);

void tile_sequence_destructor(void* data);

GenericNode* create_tiles_tree_from_tiles_ids(GenericNode *const all_tiles,
                                              const unsigned int* const ids,
                                              const unsigned int count);

TilesSequence* create_tiles_sequence_from_tile_ids(GenericNode *const tiles,
                                                   const unsigned int* const ids,
                                                   const unsigned int count);

TileGroupsSequence* create_tiles_groups_sequence_from_ids(GenericNode *const all_tiles,
                                                          const unsigned int* const leader_tiles_ids,
                                                          const unsigned int count);


char choose_best_result(const NodeResult left_node_result, const NodeResult right_node_result);

#endif // CLUSTER_UTILS_H
