#ifndef REDUCE_UTILS_H
#define REDUCE_UTILS_H

#include "tile_utils.h"
#include "generic_avl_tree.h"
#include "params.h"

typedef GenericNode reduce_results_t;

#define reduce_results_free(reduce_results) destroy_tree(reduce_results, &free)


reduce_results_t* reduce_tiles(Tile* const* const tiles_array,
                               const unsigned int count,
                               const tile_reducer_params params);


#endif // REDUCE_UTILS_H
