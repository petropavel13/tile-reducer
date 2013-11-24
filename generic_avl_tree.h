#ifndef GENERIC_AVL_TREE_H
#define GENERIC_AVL_TREE_H

#include <stdlib.h>

typedef struct GenericNode {
    unsigned long key;
    void* data;

    struct GenericNode* left;
    struct GenericNode* right;
    unsigned char height;
} GenericNode;

typedef struct TreeInfo {
    void (*data_destructor) (void* data);
} TreeInfo;


GenericNode* create_node(const unsigned long key, void *const data);

void recalc_height(GenericNode* const p);

GenericNode* rotate_right(GenericNode* const p);
GenericNode* rotate_left(GenericNode* const q);

GenericNode* balance(GenericNode* const p);


GenericNode* find(GenericNode *const node, const unsigned long key);

GenericNode* insert(GenericNode *p, const unsigned long key, void *const data);

GenericNode* find_min(GenericNode *const p);

GenericNode* remove_min(GenericNode *const p);

GenericNode* remove_node(GenericNode* const p, unsigned long key, const TreeInfo *const tree_info);

void destroy_tree(GenericNode* const root_node, const TreeInfo* const tree_info);

void calc_elements_count(const GenericNode* const node, unsigned long* const count);

void avl_shallow_copy(const GenericNode* const src, GenericNode* dest);

static inline unsigned char get_height(const GenericNode* const p)
{
    return p != NULL ? p->height : 0;
}

static inline int calc_bfactor(const GenericNode* const p)
{
    return get_height(p->right) - get_height(p->left);
}

static inline unsigned long make_key(const unsigned int x, const unsigned int y) {
    unsigned long max = x;
    unsigned long min = y;

    if(max < min) {
        const unsigned int tmp = max;
        max = min;
        min = tmp;
    }

    return max * max + max + min;
}

#endif // GENERIC_AVL_TREE_H
