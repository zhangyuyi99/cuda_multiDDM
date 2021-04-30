#include <string>

#ifndef _AZIMUTHAL_AVERAGE_
#define _AZIMUTHAL_AVERAGE_

void buildAzimuthMask(bool *d_mask_out, int *h_pixel_counts, float *q_arr, int q_count, float q_tolerance, int w, int h);

void analyseFFTDevice(std::string filename,
					  float *d_data_in,
					  bool *d_mask,
					  int *h_px_count,
					  float norm_factor,
					  int *tau_arr, int tau_count,
					  float *q_label_arr, int q_count,
					  int tile_count,
					  int width,
					  int height,
					  int fps);

void analyseFFTHost(std::string filename,
					float *d_data_in,
					float norm_factor,
					float *q_arr, int q_count,
					int *tau_arr, int tau_count,
					float q_tolerance,
					int w, int h,
					int tile_index,
					int fps);


#endif
