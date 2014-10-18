#ifndef IMAGE_GENERATION_UTILS_H
#define IMAGE_GENERATION_UTILS_H

unsigned char* generate_white_image(const unsigned int w, const unsigned int h);
unsigned char* generate_black_image(const unsigned int w, const unsigned int h);

unsigned char* generate_white_black_crossing_squares_image(const unsigned int w, const unsigned int h);

unsigned char* generate_white_black_image(const unsigned int w,
                                          const unsigned int h,
                                          const unsigned int white_pixels_count,
                                          const unsigned int black_pixels_count);

#endif // IMAGE_GENERATION_UTILS_H
