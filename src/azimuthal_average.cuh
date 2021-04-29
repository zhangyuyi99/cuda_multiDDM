#include <string>

#ifndef _AZIMUTHAL_AVERAGE_
#define _AZIMUTHAL_AVERAGE_

void buildAzimuthMask(bool *d_mask_out, int *h_pixel_counts, float *q_arr, int q_count, float q_tolerance, int w, int h);

void multiAverageDevice(std::string filename,
                       float **d_data,
                       bool **d_mask,
                       int *h_pixel_count,
                       float norm_factor,
                       int tau_count,
                       int *tau_vector,
                       int q_count,
                       float *q_vector,
                       int scale_count,
                       int *scale_vector,
                       int fps);

void analyseFFTDevice(std::string filename,
                      float *d_data,
                      bool *d_mask,
                      int *h_px_count,
                      float norm_factor,
                      int tau_count,
                      int *tau_vector,
                      int q_count,
                      float *q_vector,
                      int tile_count,
                      int width,
                      int height,
                      int fps);

void analyseFFTHost(std::string filename,
                    float *d_in,
                    float norm_factor,
                    float *q_vector,
                    int q_count,
                    int *tau_vector,
                    int tau_count,
                    float tolerance,
                    int width,
                    int tile_count,
                    int fps);


#endif
