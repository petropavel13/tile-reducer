#include "generic_avl_tree.h"

GenericNode* create_node(unsigned long key, void *data) {
    GenericNode* new_node = malloc(sizeof(GenericNode));
    new_node->key = key;
    new_node->data = data;

    new_node->left = NULL;
    new_node->right = NULL;
    new_node->height = 0;

    return new_node;
}

void recalc_height(GenericNode* const p)
{
    const unsigned char hl = get_height(p->left);
    const unsigned char hr = get_height(p->right);
    p->height = (hl > hr ? hl : hr) + 1;
}

GenericNode* rotate_right(GenericNode* p)
{
    GenericNode* const q = p->left;
    p->left = q->right;
    q->right = p;
    recalc_height(p);
    recalc_height(q);
    return q;
}

GenericNode* rotate_left(GenericNode* q)
{
    GenericNode* const p = q->right;
    q->right = p->left;
    p->left = q;
    recalc_height(q);
    recalc_height(p);
    return p;
}

GenericNode* balance(GenericNode* const p)
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

GenericNode* find(GenericNode *node, unsigned long key) {
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

GenericNode* insert(GenericNode* const p, unsigned long key, void* data) // вставка ключа k в дерево с корнем p
{
    if( p == NULL ) {
        return create_node(key, data);
    }

    if( key < p->key ) {
        p->left = insert(p->left, key, data);
    }
    else {
        p->right = insert(p->right, key, data);
    }

    return balance(p);
}

GenericNode* find_min(GenericNode* p) // поиск узла с минимальным ключом в дереве p
{
    return p->left != NULL ? find_min(p->left) : p;
}

GenericNode* remove_min(GenericNode* const p) // удаление узла с минимальным ключом из дерева p
{
    if(p->left == NULL) {
        return p->right;
    }

    p->left = remove_min(p->left);

    return balance(p);
}

GenericNode* remove_node(GenericNode* const p, unsigned long key, const TreeInfo* const tree_info)
{
    if( p == NULL ) {
        return NULL;
    }

    if( key < p->key ) {
        p->left = remove_node(p->left, key, tree_info);
    }
    else if( key > p->key ) {
        p->right = remove_node(p->right, key, tree_info);
    }
    else //  k == p->key
    {
        GenericNode* q = p->left;
        GenericNode* const r = p->right;

        if(tree_info->data_destructor != NULL) { // NULL means data must be untouched
            tree_info->data_destructor(p->data);
        }

        free(p);

        if( r == NULL ) {
            return q;
        }

        GenericNode* const min = find_min(r);
        min->right = remove_min(r);
        min->left = q;

        return balance(min);
    }

    return balance(p);
}

void destroy_tree(GenericNode* root_node, const TreeInfo* const tree_info) {
    GenericNode* node = root_node;

    if(node != NULL) {
        destroy_tree(node->left, tree_info);
        destroy_tree(node->right, tree_info);

        if(tree_info->data_destructor != NULL) { // NULL means data must be untouched
            tree_info->data_destructor(root_node->data);
        }

        free(node);
    }
}

void calc_elements_count(const GenericNode* const node, unsigned long *const count) {
    if(node != NULL) {
        (*count)++;
        calc_elements_count(node->left, count);
        calc_elements_count(node->right, count);
    }
}
