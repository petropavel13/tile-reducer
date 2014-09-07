#include "fs_utils.h"

#include <string.h>
#include <dirent.h>
#include <stdio.h>
#include <sys/stat.h>

void read_files_in_folder_recursive(const char* const absolute_path, void* const context_callback, void (*callback)(const char*, void* const)) {
    DIR* dir = NULL;
    struct dirent *entry;

    if((dir = opendir(absolute_path)) != NULL) {
        while ((entry = readdir(dir)) != 0) {
            struct stat s;
            stat(entry->d_name, &s);

            if(S_ISDIR(s.st_mode)) {
                if(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                    continue;

                char file_path[strlen(absolute_path) + strlen(entry->d_name + 1)];
                sprintf(file_path, "%s%s/", absolute_path, entry->d_name);

                read_files_in_folder_recursive(file_path, context_callback, callback);
            } else {
                char file_path[strlen(absolute_path) + strlen(entry->d_name) + 1];
                sprintf(file_path, "%s%s", absolute_path, entry->d_name);

                callback(file_path, context_callback);
            }
        }

        closedir(dir);
    }
}
