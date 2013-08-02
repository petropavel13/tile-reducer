#include "avl_tree.h"

TreeNode* createNode(unsigned long key, unsigned short int diff_pixels) {
    TreeNode* new_node = malloc(sizeof(TreeNode));
    new_node->key = key;
    new_node->diff_pixels = diff_pixels;

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

TreeNode* find(TreeNode *node, unsigned long key) {
    if( node == NULL ) {
        return NULL;
    }

    if( key < node->key ) {
        return find(node->left, key);
    }
    else if( key > node->key ) {
        return find(node->right, key);
    } else {
        return node;
    }
}

TreeNode* insert(TreeNode* const p, unsigned long key, unsigned short int diff_pixels) // вставка ключа k в дерево с корнем p
{
    if( p == NULL ) {
        return createNode(key, diff_pixels);
    }

    if( key < p->key ) {
        p->left = insert(p->left, key, diff_pixels);
    }
    else {
        p->right = insert(p->right, key, diff_pixels);
    }

    return balance(p);
}

TreeNode* find_min(TreeNode* p) // поиск узла с минимальным ключом в дереве p
{
    return p->left != NULL ? find_min(p->left) : p;
}

TreeNode* remove_min(TreeNode* const p) // удаление узла с минимальным ключом из дерева p
{
    if(p->left == NULL) {
        return p->right;
    }

    p->left = remove_min(p->left);

    return balance(p);
}

TreeNode* remove_node(TreeNode* const p, unsigned long key) // удаление ключа k из дерева p
{
    if( p == NULL ) {
        return NULL;
    }

    if( key < p->key ) {
        p->left = remove_node(p->left, key);
    }
    else if( key > p->key ) {
        p->right = remove_node(p->right, key);
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

void remove_node_fast(TreeNode* const p) {
    TreeNode* q = p->left;
    TreeNode* const r = p->right;

    free(p);

    if( r == NULL ) {
        return;
    }

    TreeNode* const min = find_min(r);
    min->right = remove_min(r);
    min->left = q;
}
