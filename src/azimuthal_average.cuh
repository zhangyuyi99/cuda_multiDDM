#include <string>

#ifndef _AZIMUTHAL_AVERAGE_
#define _AZIMUTHAL_AVERAGE_

void buildAzimuthMask(bool *d_mask_out,
					  int *h_pixel_counts,
					  float *q_arr, int q_count,
					  float q_tolerance,
					  int w, int h);

float * analyseFFTDevice(float *d_data_in,
						 bool *d_mask,
						 int *h_pixel_counts,
						 float normalisation_factor,
						 int tau_count,
						 int q_count,
						 int tile_count,
						 int w, int h);

float * analyseFFTHost(float *d_data_in,
					float norm_factor,
					float *q_arr, int q_count,
					int *tau_arr, int tau_count,
					float q_tolerance,
					int w, int h,
					int tile_index);

void writeIqtToFile(std::string filename,
					float *ISF,
					float *lambda_arr, int lamda_count,
					int   *tau_arr,	   int tau_count,
					int fps);

#endif
