#include "avl_tree.h"

TreeNode* createNode(unsigned long key) {
    TreeNode* new_node = malloc(sizeof(TreeNode));
    new_node->key = key;
    new_node->left = NULL;
    new_node->right = NULL;
    new_node->height = 0;

    return new_node;
}

void recalc_height(TreeNode* const p)
{
    const unsigned char hl = height(p->left);
    const unsigned char hr = height(p->right);
    p->height = (hl > hr ? hl : hr) + 1;
}

TreeNode* rotate_right(TreeNode* p) // правый поворот вокруг p
{
    TreeNode* const q = p->left;
    p->left = q->right;
    q->right = p;
    recalc_height(p);
    recalc_height(q);
    return q;
}

TreeNode* rotate_left(TreeNode* q) // левый поворот вокруг q
{
    TreeNode* const p = q->right;
    q->right = p->left;
    p->left = q;
    recalc_height(q);
    recalc_height(p);
    return p;
}

TreeNode* balance(TreeNode* const p) // балансировка узла p
{
    recalc_height(p);

    if( calc_bfactor(p) == 2 )
    {
        if( calc_bfactor(p->right) < 0 ) {
            p->right = rotate_right(p->right);
        }

        return rotate_left(p);
    }

    if( calc_bfactor(p) == -2 )
    {
        if( calc_bfactor(p->left) > 0  ) {
            p->left = rotate_left(p->left);
        }

        return rotate_right(p);
    }

    return p; // балансировка не нужна
}

TreeNode* insert(TreeNode* const p, unsigned long key) // вставка ключа k в дерево с корнем p
{
    if( p == NULL ) {
        return createNode(key);
    }

    if( key < p->key ) {
        p->left = insert(p->left,key);
    }
    else {
        p->right = insert(p->right,key);
    }

    return balance(p);
}

TreeNode* find_min(TreeNode* p) // поиск узла с минимальным ключом в дереве p
{
    return p->left != NULL ? find_min(p->left) : p;
}

TreeNode* remove_min(TreeNode* const p) // удаление узла с минимальным ключом из дерева p
{
    if(p->left == 0) {
        return p->right;
    }

    p->left = remove_min(p->left);

    return balance(p);
}

TreeNode* remove(TreeNode* const p, unsigned long key) // удаление ключа k из дерева p
{
    if( p == NULL ) {
        return 0;
    }

    if( key < p->key ) {
        p->left = remove(p->left, key);
    }
    else if( key > p->key ) {
        p->right = remove(p->right, key);
    }
    else //  k == p->key
    {
        TreeNode* q = p->left;
        TreeNode* const r = p->right;

        free(p);

        if( r == NULL ) {
            return q;
        }

        TreeNode* const min = find_min(r);
        min->right = remove_min(r);
        min->left = q;

        return balance(min);
    }

    return balance(p);
}
