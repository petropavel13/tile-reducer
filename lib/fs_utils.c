#include "fs_utils.h"

#include <string.h>
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>

void read_files_in_folder_recursive(const char* const absolute_path,
                                    void* const callback_context,
                                    void (*callback)(const char*, void* const)) {
    DIR* dir = NULL;
    struct dirent *entry;

    if((dir = opendir(absolute_path)) != NULL) {
        while ((entry = readdir(dir)) != 0) {
            const unsigned int path_len = strlen(absolute_path) + strlen(entry->d_name) + 1 /* '/' */ + 1 /* '\0' */;

            char file_path[path_len];
            snprintf(file_path, path_len, "%s%s", absolute_path, entry->d_name);

            struct stat s;
            stat(file_path, &s);

            if(S_ISDIR(s.st_mode)) {
                if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                    continue;

                file_path[path_len - 2] = '/';
                file_path[path_len - 1] = '\0';

                read_files_in_folder_recursive(file_path, callback_context, callback);
            } else {
                callback(file_path, callback_context);
            }
        }

        closedir(dir);
    }
}
