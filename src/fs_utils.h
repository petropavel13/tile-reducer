#ifndef FS_UTILS_H
#define FS_UTILS_H

void read_files_in_folder_recursive(const char* const absolute_path, void* const context_callback, void (*callback)(const char*, void* const));

#endif // FS_UTILS_H
