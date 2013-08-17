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

static inline GroupElement* get_element_with_index(const GroupElement* elem, unsigned int index) {
    unsigned int current = 0;
    GroupElement* temp = elem->first;

    while ((current++ < index) && (temp != NULL)) {
        temp = temp->next;
    }

    return temp;
}

static inline SelectedPoint* get_selected_point_for_branch(const PathPoint* branch) {
    SelectedPoint* sp = branch->selected_point;

    const PathPoint* temp = branch;

    while(temp != NULL && sp == NULL) {
        sp = temp->selected_point;
        temp = temp->parent;
    }

    return sp;
}

char choose_best(const PathPoint* left_path, const PathPoint* right_path);

#endif // CLUSTER_UTILS_H
