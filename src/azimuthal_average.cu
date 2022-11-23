//////////////////////////////////////
//  Reduction code is based heavily on the reduction_example from Nvidia's CUDA SDK examples
//  See "Optimizing parallel reduction in CUDA" - M. Harris for more details
//  some tweaks in regard to adding Boolean mask made
//////////////////////////////////////

#include <string>
#include <iostream>
#include <fstream>

#include "constants.hpp"
#include "debug.hpp"
#include "azimuthal_average_kernel.cuh"

inline unsigned int nextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

///////////////////////////////////////////////////////
//	Writes ISF(lambda, tau) to file. The format of the
//	output is described in detail in the documentation.
///////////////////////////////////////////////////////
void writeIqtToFile(std::string filename,
					float *ISF,
					float *lambda_arr, int lamda_count,
					int   *tau_arr,	   int tau_count,
					float fps) {

    std::ofstream out_file(filename); // attempt to open file

    if (out_file.is_open()) {
    	// lambda - values
    	for (int lidx = 0; lidx < lamda_count; lidx++) {
    		out_file << lambda_arr[lidx] << " ";
    	}

    	out_file << "\n";

    	// tau - values
    	for (int ti = 0; ti < tau_count; ti++) {
    		out_file << static_cast<float>(tau_arr[ti]) / fps << " ";
    	}

    	out_file << "\n";

    	// I(lambda, tau) - values
		for (int li = 0; li < lamda_count; li++) {
	    	for (int ti = 0; ti < tau_count; ti++) {
	    		out_file << ISF[li * tau_count + ti] << " ";
	    	}
	    	out_file << "\n";
		}

		out_file.close();
		verbose("I(lambda, tau) written to %s\n", filename.c_str());

    } else {
		fprintf(stderr, "[Out Error] Unable to open %s.\n", filename.c_str());
		exit(EXIT_FAILURE);
    }
}

// Host analysis

///////////////////////////////////////////////////////
//	This function performs azimuthal averaging on the
//	host (i.e. CPU), in almost all cases this is far slower
//	than using the GPU, included for completeness.
//  Returns ISF
///////////////////////////////////////////////////////
float * analyseFFTHost(float *d_data_in,
					float norm_factor,
					float *q_arr, int q_count,
					int *tau_arr, int tau_count,
					float q_tolerance,
					int w, int h,
					int tile_index) {

	float * ISF = new float[tau_count * q_count];

	float q2_arr[q_count]; // array containing squared q-values
	for (int i = 0; i < q_count; i++)
		q2_arr[i] = q_arr[i] * q_arr[i];

	int element_count = (w/2 + 1) * h; // number of elements in mask

	// pre-calc some values
	float tol2 = q_tolerance * q_tolerance;
	int half_w = w / 2;
	int half_h = h / 2;

	int x_shift, y_shift;
	float r2, r2q2_ratio;

    for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
		for (int q_idx = 0; q_idx < q_count; q_idx++) {
			float val = 0;
			float px_count = 0;

			for (int x = 0; x < (w/2 + 1); x++) {
				for (int y = 0; y < h; y++) {
					// manual FFT shift
					x_shift = (x + half_w) % w;
					y_shift = (y + half_h) % h;

					// distance from centre
					x_shift -= half_w;
					y_shift -= half_h;

					r2 = x_shift * x_shift + y_shift * y_shift;
					r2q2_ratio = r2 / q2_arr[q_idx];

					// element true if r in range [1.0 q, q_tolerance * q]
					if (1 <= r2q2_ratio && r2q2_ratio <= tol2) {
						val += d_data_in[element_count * tau_idx * tile_index + y * (w/2 + 1) + x];
						px_count += 1.0;
					}

		    		val *= 2; // account for symmetry
		            val /= px_count;
		            val /= norm_factor;

		            ISF[q_idx * tau_count + tau_idx] = val;
				}
			}
		}
    }

    return ISF;
}

// Device analysis

///////////////////////////////////////////////////////
//	This function builds azimuthal boolean pixel masks
//	based on given input parameters. Masks are built on
//	host and copied to given device memory location.
///////////////////////////////////////////////////////
// buildAzimuthMask(d_masks, h_pixel_counts, q_pixel_radius, lambda_count, mask_tolerance, scale, scale);
void buildAzimuthMask(bool *d_mask_out,
					  int *h_pixel_counts,
					  float *q_arr, int q_count,
					  float q_tolerance,
					  int w, int h) {

	float q2_arr[q_count]; // array containing squared q-values
	for (int i = 0; i < q_count; i++)
		q2_arr[i] = q_arr[i] * q_arr[i];

	int element_count = (w/2 + 1) * h; // number of elements in mask
	bool *h_mask = new bool[element_count * q_count];

	// pre-calc some values
	float tol2 = q_tolerance * q_tolerance;
	int half_w = w / 2;
	int half_h = h / 2;

	int x_shift, y_shift;
	float r2, r2q2_ratio;

	bool px;
	for (int q_idx = 0; q_idx < q_count; q_idx++) {
		h_pixel_counts[q_idx] = 0;  // TODO: h_pixel_counts[(s-1)*q_count+q_idx] = 0;

		for (int x = 0; x < (w/2 + 1); x++) {
			for (int y = 0; y < h; y++) {
				// manual FFT shift
				x_shift = (x + half_w) % w;
				y_shift = (y + half_h) % h;  // TODO: why manual shift?

				// distance from centre
				x_shift -= half_w;
				y_shift -= half_h;

				r2 = x_shift * x_shift + y_shift * y_shift;
				r2q2_ratio = r2 / q2_arr[q_idx];

				// element true if r in range [1.0 q, q_tolerance * q]
				px = (1 <= r2q2_ratio) && (r2q2_ratio <= tol2);
                if (px) h_pixel_counts[q_idx] += 1; // TODO: h_pixel_counts[(s-1)*q_count+q_idx] = 0;
                h_mask[q_idx * element_count + y * (w/2 + 1) + x] = px;
			}
		}

		if (h_pixel_counts[q_idx] == 0) {
			verbose("[Mask Generation] q: %f, (#q: %d) has zero mask pixels for scale %d x %d\n", q_arr[q_idx], q_idx, w, h);
		}

    }
	// Copy mask onto GPU
    gpuErrorCheck(cudaMemcpy(d_mask_out, h_mask, sizeof(bool) * element_count * q_count, cudaMemcpyHostToDevice));
    delete h_mask;

}


///////////////////////////////////////////////////////
// Code to perform masked (GPU) reduction of ISF
// For now just use whole mask, in future could investigate
// Performing 2 reductions on (w / 2) * (h / 2) as this would
// most likely be a power of 2, for which reduction most optimised
///////////////////////////////////////////////////////
float * analyseFFTDevice(float *d_data_in,
					  bool *d_mask,
					  int *h_px_count,
					  float norm_factor,
					  int tau_count,
					  int q_count,
					  int tile_count,
					  int w, int h) {

	int n = (w / 2 + 1) * h;

	// Compute the number of threads and blocks to use for the reduction kernel

	int threads = (n < BLOCKSIZE * 2) ? nextPow2((n + 1) / 2) : BLOCKSIZE;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);

	blocks = (64 < blocks) ? 64 : blocks;

	float *d_intermediateSums;
	float *h_intermediateSums = new float[blocks];

	gpuErrorCheck(cudaMalloc((void **)&d_intermediateSums, sizeof(float) * blocks));

	float * ISF = new float[tau_count * q_count]();
	for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
		for (int q_idx = 0; q_idx < q_count; q_idx++) {

			float val = 0;

			if (h_px_count[q_idx] != 0) {

				// execute the kernel
				maskReduce<float>(n, threads, blocks, d_data_in + n*tau_idx*tile_count, d_mask + n*q_idx, d_intermediateSums);

				// copy result from device to host
				gpuErrorCheck(cudaMemcpy(h_intermediateSums, d_intermediateSums, blocks * sizeof(float), cudaMemcpyDeviceToHost));

				// sum partial sums from each block on CPU
				for (int i = 0; i < blocks; i++) {
					val += h_intermediateSums[i];
				}

				val *= 2; 										// account for symmetry
				val /= static_cast<float>(h_px_count[q_idx]);	// divide by number of pixels
				val *= norm_factor;								// normalise
			}

			ISF[q_idx * tau_count + tau_idx] = val;
		}
	}

	cudaDeviceSynchronize();

	return ISF;
}


