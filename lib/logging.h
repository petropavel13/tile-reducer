#ifndef LOGGING_H
#define LOGGING_H

#include <log4c.h>

#ifdef __cplusplus
#define TILE_REDUCER_EXPORT extern "C"
#else
#define TILE_REDUCER_EXPORT extern
#endif

#ifndef NO_LOG

TILE_REDUCER_EXPORT log4c_category_t* tile_reducer_log_category;

TILE_REDUCER_EXPORT void tile_reducer_log_init();

#define tile_reducer_log_fini() log4c_fini()

#define tile_reducer_log(priority, format, ...) log4c_category_log(tile_reducer_log_category, priority, format , ## __VA_ARGS__)
#define tile_reducer_log_debug(format, ...) log4c_category_log(tile_reducer_log_category, LOG4C_PRIORITY_DEBUG, format , ## __VA_ARGS__)
#define tile_reducer_log_info(format, ...) log4c_category_log(tile_reducer_log_category, LOG4C_PRIORITY_INFO, format , ## __VA_ARGS__)
#define tile_reducer_log_warn(format, ...) log4c_category_log(tile_reducer_log_category, LOG4C_PRIORITY_WARN, format , ## __VA_ARGS__)
#define tile_reducer_log_error(format, ...) log4c_category_log(tile_reducer_log_category, LOG4C_PRIORITY_ERROR, format , ##  __VA_ARGS__)

#define tile_reducer_current_filename_and_line_string_declare char __outbuf[256]; char* const __outbuf_ref = &__outbuf[0];
#define tile_reducer_current_filename_and_line_string __outbuf_ref
#define tile_reducer_current_filename_and_line_to_string() sprintf(__outbuf_ref, "%s:%d", __FILE__, __LINE__)

#else

#define tile_reducer_log_init()

#define tile_reducer_log_fini()

#define tile_reducer_log(...)
#define tile_reducer_log_debug(...)
#define tile_reducer_log_info(...)
#define tile_reducer_log_warn(...)
#define tile_reducer_log_error(...)

#define tile_reducer_current_filename_and_line_string_declare
#define tile_reducer_current_filename_and_line_string

#define tile_reducer_current_filename_and_line_to_string()
#endif

#endif // LOGGING_H
