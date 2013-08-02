#ifndef AVL_TREE_H
#define AVL_TREE_H

#include <stdlib.h>

typedef struct TreeNode // структура для представления узлов дерева
{
    unsigned long key;
    unsigned short int diff_pixels;
    unsigned char height;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

TreeNode* createNode(unsigned long key, unsigned short diff_pixels);

void recalc_height(TreeNode* const p);

TreeNode* rotate_right(TreeNode* p);
TreeNode* rotate_left(TreeNode* q);

TreeNode* balance(TreeNode* const p);


TreeNode* find(TreeNode* node, unsigned long key);

TreeNode* insert(TreeNode* const p, unsigned long key, unsigned short int diff_pixels);

TreeNode* find_min(TreeNode *p);

TreeNode* remove_min(TreeNode *const p);

TreeNode* remove_node(TreeNode* const p, unsigned long key);

void remove_node_fast(TreeNode* const p);


static inline unsigned char height(const TreeNode* const p)
{
    return p ? p->height : 0;
}

static inline int calc_bfactor(const TreeNode* const p)
{
    return height(p->right) - height(p->left);
}



#endif // AVL_TREE_H
