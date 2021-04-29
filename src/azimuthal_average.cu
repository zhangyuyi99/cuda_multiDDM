// Code draws on Nvidia SDK reduction_example

#include <string>
#include <iostream>
#include <fstream>

#include "constants.hpp"
#include "debug.hpp"
#include "reduction.h"


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
//	Writes I(q, tau) to file. The format of the
//	output is described in detail in the documentation.
///////////////////////////////////////////////////////
void writeIqtToFile(std::string filename,
					float *iqtau,
					float *q_arr, int q_count,
					int *tau_arr, int tau_count,
					int fps) {

    std::ofstream out_file(filename); // attempt to open file

    if (out_file.is_open()) {
    	// q - values
    	for (int qi = 0; qi < q_count; qi++) {
    		out_file << q_arr[qi] << " ";
    	}

    	out_file << "\n";

    	// tau - values
    	for (int ti = 0; ti < tau_count; ti++) {
    		out_file << static_cast<float>(tau_arr[ti]) / static_cast<float>(fps) << " ";
    	}

    	out_file << "\n";

    	// I(q, tau) - values
		for (int qi = 0; qi < q_count; qi++) {
	    	for (int ti = 0; ti < tau_count; ti++) {
	    		out_file << iqtau[qi * tau_count + ti] << " ";
	    	}
	    	out_file << "\n";
		}

		out_file.close();
		verbose("I(Q, tau) written to %s\n", filename.c_str());
    } else {
		fprintf(stderr, "[Out Error] Unable to open %s.\n", filename.c_str());
		exit(EXIT_FAILURE);
    }
}

///////////////////////////////////////////////////////
//	This function performs azimuthal averaging on the
//	host (i.e. CPU), in almost all cases this is far slower
//	than using the GPU, included for completeness.
///////////////////////////////////////////////////////
void analyseFFTHost(std::string filename,
					float *d_data_in,
					float norm_factor,
					float *q_arr, int q_count,
					int *tau_arr, int tau_count,
					float q_tolerance,
					int w, int h,
					int tile_index,
					int fps) {

	float * iqtau = new float[tau_count * q_count];

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

					iqtau[q_idx * tau_count + tau_idx] = val;
				}
			}
		}
    }

    // Finally write I(q, tau) to file
    writeIqtToFile(filename, iqtau, q_arr, q_count, tau_arr, tau_count, fps);
}

///////////////////////////////////////////////////////
//	This function builds azimuthal boolean pixel masks
//	based on given input parameters. Masks are built on
//	host and copied to given device memory location.
///////////////////////////////////////////////////////
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
		h_pixel_counts[q_idx] = 0;

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
				px = (1 <= r2q2_ratio) && (r2q2_ratio <= tol2);
                if (px) h_pixel_counts[q_idx] += 1;
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

///////// NVIDIA ////////

// For now just use whole mask, in future could investigate
// Performing 2 reductions on (w / 2) * (h / 2) as this would
// most likely be a power of 2, for which reduction most optimised
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
					  int fps) {

	// TODO: move to boarder function
	// get device capability, to avoid block/grid size exceed the upper bound
	cudaDeviceProp prop;
	int device;
	gpuErrorCheck(cudaGetDevice(&device));
	gpuErrorCheck(cudaGetDeviceProperties(&prop, device));

	int n = (width / 2 + 1) * height;

	// Compute the number of threads and blocks to use for the given reduction
	// kernel For the kernels >= 3, we set threads / block to the minimum of
	// maxThreads and n/2.

	int threads = (n < BLOCKSIZE * 2) ? nextPow2((n + 1) / 2) : BLOCKSIZE;
	int blocks = (n + (threads * 2 - 1)) / (threads * 2);

	if ((float)threads * blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
		printf("[Reduction] Image is too large.\n");
		exit(EXIT_FAILURE);
	}

	if (blocks > prop.maxGridSize[0]) {
		printf("[Reduction] Grid size <%d> exceeds the device capability <%d>, set block size as "
				"%d (original %d)\n", blocks, prop.maxGridSize[0], threads * 2, threads);

		blocks /= 2;
		threads *= 2;
	}

	blocks = (64 < blocks) ? 64 : blocks;

	float *d_intermediateSums;
	float *h_intermediateSums = new float[blocks];

	gpuErrorCheck(cudaMalloc((void **)&d_intermediateSums, sizeof(float) * blocks));


	float * iq_tau = new float[tau_count * q_count]();
	for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
		for (int q_idx = 0; q_idx < q_count; q_idx++) {

			float val = 0;

			if (h_px_count[q_idx] != 0) {

				// execute the kernel
				maskReduce<float>(n, threads, blocks, d_data_in + n*tau_idx*tile_count, d_mask + n*q_idx, d_intermediateSums);

				// check if kernel execution generated an error
				//getLastCudaError("Kernel execution failed");

				// sum partial sums from each block on CPU TODO can do this on device too
				// copy result from device to host
				gpuErrorCheck(cudaMemcpy(h_intermediateSums, d_intermediateSums, blocks * sizeof(float),
							 cudaMemcpyDeviceToHost));

				for (int i = 0; i < blocks; i++) {
					val += h_intermediateSums[i];
				}

				val *= 2; // account for symmetry
				val /= static_cast<float>(h_px_count[q_idx]);
				val /= norm_factor;
			}

        	iq_tau[q_idx * tau_count + tau_idx] = val;
		}
	}

	cudaDeviceSynchronize();

    // Finally write I(q, tau) to file
    writeIqtToFile(filename, iq_tau, q_label_arr, q_count, tau_arr, tau_count, fps);


}

///////// NVIDIA END ////////


