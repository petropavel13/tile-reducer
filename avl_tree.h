#ifndef AVL_TREE_H
#define AVL_TREE_H

#include <stdlib.h>

typedef struct TreeNode // структура для представления узлов дерева
{
    unsigned long key;    
    struct TreeNode* left;
    struct TreeNode* right;
    unsigned short int diff_pixels;
    unsigned char height;
} TreeNode;

TreeNode* create_node(unsigned long key, unsigned short diff_pixels);

void recalc_height(TreeNode* const p);

TreeNode* rotate_right(TreeNode* p);
TreeNode* rotate_left(TreeNode* q);

TreeNode* balance(TreeNode* const p);


TreeNode* find(TreeNode* node, unsigned long key);

TreeNode* insert(TreeNode* const p, unsigned long key, unsigned short int diff_pixels);

TreeNode* find_min(TreeNode *p);

TreeNode* remove_min(TreeNode *const p);

TreeNode* remove_node(TreeNode* const p, unsigned long key);


static inline unsigned char get_height(const TreeNode* const p)
{
    return p != NULL ? p->height : 0;
}

static inline int calc_bfactor(const TreeNode* const p)
{
    return get_height(p->right) - get_height(p->left);
}



#endif // AVL_TREE_H
