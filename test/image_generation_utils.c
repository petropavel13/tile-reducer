#include "image_generation_utils.h"

#include <stdlib.h>

unsigned char* generate_white_image(const unsigned int w, const unsigned int h) {
    unsigned char* const image = malloc(sizeof(char) * w * h * 4);

    for (unsigned int row = 0, row_idx = 0; row < h; ++row, row_idx = row * w * 4) {
        for (unsigned int col = 0, pixel_idx = row_idx; col < w; ++col, pixel_idx += 4) {
            image[pixel_idx + 0] = 255;
            image[pixel_idx + 1] = 255;
            image[pixel_idx + 2] = 255;
            image[pixel_idx + 3] = 255;
        }
    }

    return image;
}

unsigned char* generate_black_image(const unsigned int w, const unsigned int h) {
    unsigned char* const image = malloc(sizeof(char) * w * h * 4);

    for (unsigned int row = 0, row_idx = 0; row < h; ++row, row_idx = row * w * 4) {
        for (unsigned int col = 0, pixel_idx = row_idx; col < w; ++col, pixel_idx += 4) {
            image[pixel_idx + 0] = 0;
            image[pixel_idx + 1] = 0;
            image[pixel_idx + 2] = 0;
            image[pixel_idx + 3] = 255;
        }
    }

    return image;
}

unsigned char* generate_white_black_crossing_squares_image(const unsigned int w, const unsigned int h) {
    unsigned char* const image = malloc(sizeof(char) * w * h * 4);

    unsigned char t_color;

    // looks's like obfuscated code, I know. sorry about that :D
    for (unsigned int row = 0, row_idx = row, e_row = row % 2; row < h; ++row, row_idx = row * w * 4, e_row = row % 2) {
        for (unsigned int col = 0, p_idx = row_idx, e_col = col % 2; col < w; ++col, p_idx += 4, e_col = col % 2) {
            t_color = (e_row ^ e_col) * 255; // 0 or 255

            image[p_idx + 0] = t_color;
            image[p_idx + 1] = t_color;
            image[p_idx + 2] = t_color;
            image[p_idx + 3] = 255;
        }
    }

    return image;
}

unsigned char* generate_white_black_image(const unsigned int w,
                                          const unsigned int h,
                                          const unsigned int white_pixels_count,
                                          const unsigned int black_pixels_count) {
    unsigned char* const image = malloc(sizeof(char) * w * h * 4);

    unsigned char t_color = 255;

    for (unsigned int row = 0, row_idx = 0; row < h; ++row, row_idx = row * w * 4) {
        for (unsigned int col = 0, pixel_idx = row_idx; col < w; ++col, pixel_idx += 4) {
            t_color = ((pixel_idx / 4) < white_pixels_count) * 255; // 0 or 255

            image[pixel_idx + 0] = t_color;
            image[pixel_idx + 1] = t_color;
            image[pixel_idx + 2] = t_color;
            image[pixel_idx + 3] = ((pixel_idx / 4) < black_pixels_count) * 255; // 0 or 255 -> last pixels transparent
        }
    }

    return image;
}
