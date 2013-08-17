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


GenericNode* create_node(unsigned long key, void *data);

void recalc_height(GenericNode* const p);

GenericNode* rotate_right(GenericNode* p);
GenericNode* rotate_left(GenericNode* q);

GenericNode* balance(GenericNode* const p);


GenericNode* find(GenericNode* node, unsigned long key);

GenericNode* insert(GenericNode* const p, unsigned long key, void *data);

GenericNode* find_min(GenericNode *p);

GenericNode* remove_min(GenericNode *const p);

GenericNode* remove_node(GenericNode* const p, unsigned long key, const TreeInfo *const tree_info);

void destroy_tree(GenericNode* root_node, const TreeInfo* const tree_info);


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
