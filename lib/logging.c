#include "logging.h"

#include <stdio.h>

#ifndef NO_LOG

log4c_category_t* tile_reducer_log_category;

void tile_reducer_log_init() {
    if (log4c_init() != 0) {
        printf("log4c_init() failed\n");
    }

    if ((tile_reducer_log_category = log4c_category_get("tile_reducer_lib")) == NULL) {
        printf("Filed to create log4c category!\n");
    }
}

#endif

