#ifndef FS_UTILS_H
#define FS_UTILS_H

/**
 * @brief read files in folder recursive and callback on each on file
 * @param absolute_path relative paths like '~/' or '../' not accepted
 * @param context_callback
 */
void read_files_in_folder_recursive(const char* const absolute_path,
                                    void* const callback_context,
                                    void (*callback)(const char*, void* const));

#endif // FS_UTILS_H
