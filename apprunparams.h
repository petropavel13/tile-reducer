#ifndef RUNPARAMS_H
#define RUNPARAMS_H

typedef struct AppRunParams {
    unsigned short int max_diff_pixels;
    unsigned char max_num_threads;
} AppRunParams;


static inline AppRunParams make_app_run_params(const unsigned short int max_diff_pixels,
                                 const unsigned char max_num_threads) {
    AppRunParams rp = { max_diff_pixels, max_num_threads };

    return rp;
}

#endif // RUNPARAMS_H
