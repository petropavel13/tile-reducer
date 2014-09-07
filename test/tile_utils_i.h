#ifndef TILE_UTILS_I_H
#define TILE_UTILS_I_H

typedef enum TaskStatus {
    TASK_DONE,
    TASK_FAILED
} TaskStatus;


unsigned int compare_images_one_with_one_cpu(const unsigned char * const raw_left_image,
                                               const unsigned char * const raw_right_image);

TaskStatus compare_images_one_with_many_cpu(const unsigned char* const left_raw_image,
                                            const unsigned char* const right_raw_images,
                                            const unsigned int right_images_count,
                                            unsigned int* const diff_results);

#endif // TILE_UTILS_I_H
