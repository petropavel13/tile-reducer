#ifndef CLUSTER_UTILS_H
#define CLUSTER_UTILS_H

#include "tile_utils.h"
#include "db_utils.h"
#include <stdio.h>

typedef struct GroupElement {
    struct GroupElement* first;
    Tile* node;
    struct GroupElement* next;
} GroupElement;

typedef struct PathPoint {
    struct PathPoint* parent;
    struct SelectedPoint* selected_point;
    struct PathPoint* child_left;
    struct PathPoint* child_right;
} PathPoint;

typedef struct SelectedPoint {
    struct GroupElement* sequence_head;
    Tile* leader_node;
    unsigned int group_id;
    unsigned int groups_count;
    unsigned int rest_count;
} SelectedPoint;

#define PERSISTENT_GROUP_NOT_DEFINED 0

void make_persistent_groups(DbInfo* const db_info,
                            GroupElement *const tiles_sequence,
                            unsigned int total, CacheInfo *const cache_info);

void clusterize(GroupElement *const tiles_sequence,
                unsigned int total,
                unsigned int max_diff_pixels,
                unsigned int max_allowed_tiles,
                CacheInfo *const cache_info,
                DbInfo *const db_info);

PathPoint* make_group(GroupElement *const rest_tiles,
                      unsigned int total,
                      unsigned int offset,
                      PathPoint *const parent_path,
                      CacheInfo *const cache_info,
                      DbInfo *const db_info,
                      unsigned int max_diff_pixels,
                      unsigned int max_allowed_tiles,
                      unsigned int max_allowed_groups);

void delete_path(PathPoint* point, DbInfo *const db_info);

unsigned int get_count_of_sequence(const GroupElement *const head);

GroupElement* get_element_with_index(GroupElement *const search_start_element, unsigned int index);

Tile* find_tile_with_id(GroupElement *const search_start_element, unsigned int tile_id);

SelectedPoint* get_selected_point_for_branch(const PathPoint* branch);

char choose_best(const PathPoint* left_path, const PathPoint* right_path);

#endif // CLUSTER_UTILS_H
