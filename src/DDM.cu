#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <algorithm>
#include <iostream>

#include "azimuthal_average.cuh"
#include "debug.hpp"
#include "constants.hpp"
#include "video_reader.hpp"

#include "DDM_kernel.cuh"


// Function to swap two pointers
template <class T> inline void swap(T*& A, T*& B) {
    T* tmp = A;
    A = B;
    B = tmp;
}


// Function to swap three pointers, positional order important
template <class T> inline void rotateThreePtr(T*& A, T*& B, T*& C) {
    T* tmp = A;
    A = B;
    B = C;
    C = tmp;
}


///////////////////////////////////////////////////////
// If we choose to dual-stream the code then we must
// combine the FFT intensity accumulator associated with
// each stream.
///////////////////////////////////////////////////////
inline void combineAccumulators(float **d_accum_list_A,
                                float **d_accum_list_B,
                                int *scale_arr,
								int scale_count,
                                int tau_count) {

    dim3 blockDim(BLOCKSIZE);
    int main_scale = scale_arr[0];

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_arr[s];
        int tile_count = (main_scale / scale) * (main_scale / scale);
        int frame_size = (scale / 2 + 1) * scale * tile_count;

        int gridDim = ceil(frame_size / static_cast<float>(BLOCKSIZE));

        combineAccum<<<gridDim, blockDim>>>(d_accum_list_A[s], d_accum_list_B[s], tau_count, frame_size);
    }
}

// analyse_accums(scale_vector, scale_count, lambda_arr, lambda_count, tau_vector, tau_count, frames_left, mask_tolerance, file_out, d_accum_list_cur, info.fps);

///////////////////////////////////////////////////////
//	This function handles analysis of the I(q, tau)
//  accumulator. Given the inputed values of q it
//  handles calculation of the azimuthal averages.
///////////////////////////////////////////////////////
void analyse_accums(int *scale_arr,	int scale_count,
					float *lambda_arr,	int lambda_count,
					int *tau_arr,	int tau_count,
					int frames_analysed,
					float mask_tolerance,
		            std::string file_out,
		            float **accum_list,
		            int framerate) {

	int main_scale = scale_arr[0]; // the largest length-scale

	bool *d_masks; // device pointer to reference the boolean azimuthal masks

	gpuErrorCheck(cudaMalloc((void** ) &d_masks, sizeof(bool) * (main_scale / 2 + 1) * main_scale * lambda_count))

	int *h_pixel_counts = new int[lambda_count * scale_count](); // host array to hold the number of pixels in each mask

	float normalisation = 1.0 / static_cast<float>(frames_analysed);

	float *q_pixel_radius = new float[lambda_count]; // host array to hold temporary q values for each length-scale

	for (int s = 0; s < scale_count; s++) {
		int scale = scale_arr[s];
		int tile_count = (main_scale / scale) * (main_scale / scale);
		int tile_size = (scale / 2 + 1) * scale;

		for (int i = 0; i < lambda_count; i++) {
			q_pixel_radius[i] = static_cast<float>(scale) / (lambda_arr[i]); // key conversion between pixel movement and q radius
            // verbose("lambda_arr[%d]: %f\n", i, lambda_arr[i]);
            // verbose("q_pixel_radius[%d]: %f\n", i, q_pixel_radius[i]);
		}

        

		// Old format of the q-vectors assumed relative to largest scale
//        for (int i = 0; i < lambda_count; i++) {
//            lambda_arr_tmp[i] = lambda_arr[i] * (scale / static_cast<float>(main_scale));
//        }

        buildAzimuthMask(d_masks, h_pixel_counts, q_pixel_radius, lambda_count, mask_tolerance, scale, scale);

        for (int tile_idx = 0; tile_idx < tile_count; tile_idx++) { // loop through each tile i.e. I(q, tau)_tile

            std::string tmp_filename = file_out + std::to_string(scale) + "-" + std::to_string(tile_idx); //output filenames in the style ./<filename><scale>-<tile idx>

            float *d_accum_tmp = accum_list[s] + tile_size * tile_idx;

            float *ISF = analyseFFTDevice(d_accum_tmp, d_masks, h_pixel_counts, normalisation, tau_count, lambda_count, tile_count, scale, scale);

            // Finally write I(q, tau) to file
            writeIqtToFile(tmp_filename, ISF, lambda_arr, lambda_count, tau_arr, tau_count, framerate);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
//  This function handles the parsing of on-device raw (uchar) data into a float
//  array, and the multi-scale FFT of this data to a list of cufftComplex arrays.
////////////////////////////////////////////////////////////////////////////////
// parseChunk(d_ready, d_end_list, d_workspace_cur, scale_vector, scale_count, chunk_frame_count, info, FFT_plan_list, *stream_cur);
void parseChunk(unsigned char *d_raw_in,
                cufftComplex **d_fft_list_out,
                float *d_workspace,
                int *scale_arr,
				int scale_count,
                int frame_count,
                video_info_struct info,
                cufftHandle *fft_plan_list,
                cudaStream_t stream) {

    int main_scale = scale_arr[0];

    int x_dim = static_cast<int>(ceil(main_scale / static_cast<float>(BLOCKSIZE_X)));
    int y_dim = static_cast<int>(ceil(main_scale / static_cast<float>(BLOCKSIZE_Y)));

    dim3 gridDim(x_dim, y_dim);
    dim3 blockDim(BLOCKSIZE_X, BLOCKSIZE_Y);

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_arr[s];

        // mykernel<<<blocks, threads, shared_mem, stream>>>(args);
        parseBufferScalePow2<<<gridDim, blockDim, 0, stream>>>(d_raw_in, d_workspace, info.bpp, 0, info.w, info.h, info.x_off, info.y_off, scale, main_scale, frame_count);
        cufftSetStream(fft_plan_list[s], stream);

        int exe_code = cufftExecR2C(fft_plan_list[s], d_workspace, d_fft_list_out[s]);
        conditionAssert(exe_code == CUFFT_SUCCESS, "cuFFT execution failure", true);
    }
}


////////////////////////////////////////////////////////////////////////////////
//  This function handles the analysis of the FFT, i.e. handles the calculation
//  of the difference functions. Makes use of 3-section circular buffer - see project
////////////////////////////////////////////////////////////////////////////////
// analyseChunk(d_start_list, d_end_list, d_accum_list_cur, scale_count, scale_vector, leftover_frames, chunk_frame_count, frame_offset, tau_count, tau_vector, *stream_cur);
void analyseChunk(cufftComplex **d_fft_buffer1,
                  cufftComplex **d_fft_buffer2,
                  float **d_fft_accum_list,
                  int scale_count,
                  int *scale_vector,
                  int frame_count,
                  int chunk_frame_count,
                  int frame_offset,
                  int tau_count,
                  int *tau_vector,
                  cudaStream_t stream) {

    dim3 blockDim(BLOCKSIZE);

    int main_scale = scale_vector[0];

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];
        int tile_count = (main_scale / scale) * (main_scale / scale);
        int frame_size = (scale / 2 + 1) * scale * tile_count;

        int px_count = scale * scale;
        float fft_norm = 1.0f / px_count; // factor to normalise the FFT

        dim3 gridDim(static_cast<int>(ceil(frame_size / static_cast<float>(BLOCKSIZE))));

        int frames_left = chunk_frame_count - frame_offset;

        cufftComplex *tmp; // index pointer
        for (int t = 0; t < tau_count; t++) {
            if (tau_vector[t] < frames_left) { // check to see if second index is in next chunk
                tmp = d_fft_buffer1[s] + (frame_offset + tau_vector[t]) * frame_size;
            } else {
                tmp = d_fft_buffer2[s] + (tau_vector[t] - frames_left) * frame_size;
            }

            float *accum_out = d_fft_accum_list[s] + frame_size * t; // tmp pointer for position in accumulator array

            processFFT<<<gridDim, blockDim, 0, stream>>>(d_fft_buffer1[s] + frame_size * frame_offset, tmp, accum_out, fft_norm, frame_size);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
//  Main multi-DDM function
////////////////////////////////////////////////////////////////////////////////
void runDDM(std::string file_in,
            std::string file_out,
            int *tau_vector,
			int tau_count,
            float *lambda_arr,
			int lambda_count,
            int *scale_vector,
			int scale_count,
            int x_offset,
			int y_offset,
            int total_frames,
			int frame_offset,
            int chunk_frame_count,
            bool multistream,
            bool use_webcam,
            int webcam_idx,
            float mask_tolerance,
			bool use_moviefile,
			bool use_index_fps,
			bool use_explicit_fps,
			float explicit_fps,
            int dump_accum_after,
			bool benchmark_mode) {

    auto start_time = std::chrono::high_resolution_clock::now();
    verbose("[multiDDM Begin]\n");

    //////////
    ///  CUDA Check
    //////////

    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev = 0;

    // This will pick the best possible CUDA capable device
    gpuErrorCheck(cudaGetDeviceProperties(&deviceProp, dev));

    // Get information on CUDA device
    verbose("[Device Info] Device found, %d Multi-Processors, SM %d.%d compute capabilities.\n",
            deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

    //////////
    ///  Sort Parameter Arrays
    //////////

    // Can make assumptions later if tau / q / scale arrays are in known order

    std::sort(tau_vector, tau_vector + tau_count);
    std::sort(lambda_arr, lambda_arr + lambda_count);
    std::sort(scale_vector, scale_vector + scale_count, std::greater<int>());

    //////////
    ///  Parameter Check
    //////////

    verbose("Scale list:\n");
    for (int s = 0; s < scale_count; s++) {
        unsigned int scale = scale_vector[s];
        conditionAssert(!(scale == 0) && !(scale & (scale - 1)), "scales must be powers of two (> 0)", true);

        if (s < scale_count - 1)
            conditionAssert((scale_vector[s] > scale_vector[s + 1]), "scales should be descending order", true);
    }

    for (int t = 0; t < tau_count - 1; t++) {
        conditionAssert((tau_vector[t] >= 0), "Tau values should be positive", true);
        conditionAssert((tau_vector[t] < tau_vector[t + 1]), "Tau vector should be ascending order", true);
    }

    for (int q = 0; q < lambda_count - 1; q++) {
        conditionAssert((lambda_arr[q] >= 0), "q-vector values should be positive", true);
        conditionAssert((lambda_arr[q] < lambda_arr[q + 1]), "q-vector vector should be ascending order", true);
    }

    conditionAssert(mask_tolerance < 10 && mask_tolerance > 1.0,
            "mask_tolerance is likely undesired value, refer to README for more information");

    conditionAssert(lambda_arr[lambda_count-1] <= scale_vector[scale_count-1],
            "The largest q-vector should be smaller than the smallest scale.", true);

    verbose("%d tau-values.\n", tau_count);
    verbose("%d q-vector values.\n", lambda_count);

    verbose("Parameter Check Done.\n");
    //////////
    ///  Video Setup
    //////////

    // Web-cam alignment
    if (use_webcam) {
        cv::VideoCapture tmp_cap(webcam_idx);

        int main_scale = scale_vector[0];

        while(1) {
            cv::Mat tmp_frame;
            tmp_cap >> tmp_frame;

            for (int s = 0; s < scale_count; s++) {
                int scale = scale_vector[s];
                int tiles_per_side = (main_scale / scale);
                int tiles_per_frame = tiles_per_side * tiles_per_side;

                for (int t = 0; t < tiles_per_frame; t++) {
                    cv::Rect rect(x_offset + (t / tiles_per_side) * scale + s, y_offset + (t % tiles_per_side) * scale + s, scale, scale); // add s to each to help with readability
                    cv::rectangle(tmp_frame, rect, cv::Scalar(0, 255, 0));
                }
            }
            cv::imshow( "Web-cam", tmp_frame );

            char c=(char) cv::waitKey(25);
            if(c==27)
                break;
        }
        tmp_cap.release();
        cv::destroyAllWindows();
    }

    video_info_struct info;
    FILE *moviefile;
    cv::VideoCapture cap;

    if (benchmark_mode) {
    	info.w = scale_vector[0];
    	info.h = scale_vector[0];
    	info.bpp = 1;
    	info.fps = 1.0;
    } else if (use_moviefile) { // if we have a movie-file we use custom handler
        moviefile = fopen(file_in.c_str(), "rb");
        conditionAssert(moviefile != NULL, "couldn't open .movie file", true);
        info = initFile(moviefile, frame_offset);
    } else { // for other file types handle with OpenCV
        if (use_webcam) {
            cap = cv::VideoCapture(webcam_idx);
        } else {
            cap = cv::VideoCapture(file_in);
        }

        conditionAssert(cap.isOpened(), "error opening video file with openCV", true);

        info.w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        info.h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        info.fps = static_cast<float>(cap.get(cv::CAP_PROP_FPS)); // cast from double to float

        cv::Mat test_img;
        cap >> test_img;

        // Due to the difficulty in dealing with many image types, we only
        // deal with multi-channel data if image is CV_8U (i.e. uchar)
        int type = test_img.type();
        // decide if type is CV_8U. If so, (type % 8) = 0, bpp = channels; otherwise bpp = 0.
        info.bpp = (type % 8) ? 1 : test_img.channels();
        // info.bpp = 1;

        verbose("[test_img.depth - %d ]\n", test_img.depth());
        verbose("[test_img.type - %d ]\n", type);
        verbose("[test_img.channels - %d ]\n", test_img.channels());
        verbose("[bpp calculation - %d bytes per pixel]\n", info.bpp);

        // info.bpp = 8;
        verbose("[bpp after set - %d bytes per pixel]\n", info.bpp);

        if (!use_webcam)
            cap = cv::VideoCapture(file_in); // re-open so can view first frame again

        // Offset the video by frame_offset frames
        for (int i = 0; i < frame_offset; i++) {
        	cap >> test_img;
        }
    }


    if (use_index_fps) { // if flag to use frame indices as frame-rate (same as setting FPS to 1)
        info.fps = 1.0;
    }
    if (use_explicit_fps) { // if flag to explicitly specify the video frame-rate
    	info.fps = explicit_fps;
    }

    info.x_off = x_offset;
    info.y_off = y_offset;

    verbose("Video Setup Done.\n");
    //////////
    ///  Parameter check
    //////////

    conditionAssert((scale_vector[0] + info.x_off <= info.w && scale_vector[0] + info.y_off <= info.h),
            "the specified out dimensions must be smaller than actual image size", true);

    conditionAssert((tau_vector[tau_count - 1] <= chunk_frame_count),
            "the largest tau value must be smaller than number frames in a chunk", true);

    //////////
    ///  Initialise variables
    //////////

    const int buffer_frame_count = chunk_frame_count * 3;
    const int leftover_frames = total_frames % chunk_frame_count;
    const int total_chunks = total_frames / chunk_frame_count;
    const int main_scale = scale_vector[0];
    int chunks_already_parsed = 0;

    verbose("[Video info - (%d x %d), %d Frames (offset %d), %.4f FPS, %d bytes per pixel]\n", info.w, info.h, total_frames, frame_offset, info.fps, info.bpp);
    // streams

    cudaStream_t stream_1, stream_2;

    if (multistream) {
        cudaStreamCreate(&stream_1);
        cudaStreamCreate(&stream_2);
    } else {
        cudaStreamCreate(&stream_1);
    }


    verbose("Initialise Variables Done.\n");
    //////////
    ///  Memory Allocations
    //////////

    size_t total_host_memory   = 0;
    size_t total_device_memory = 0;


    // main device buffer, allocate memory in bytes
    size_t buffer_size  = sizeof(unsigned char) * buffer_frame_count * info.bpp * info.w * info.h;
    verbose("[buffer_size - %d bytes]\n", buffer_size);
    unsigned char *d_buffer;
    gpuErrorCheck(cudaMalloc((void** )&d_buffer, buffer_size));

    total_device_memory += buffer_size;

    // host buffer (multi-stream)
    size_t chunk_size  = sizeof(unsigned char) * chunk_frame_count * info.bpp * info.w * info.h;

    unsigned char *h_chunk_1;
    unsigned char *h_chunk_2;

    if (multistream) {
        gpuErrorCheck(cudaHostAlloc((void **) &h_chunk_1, chunk_size, cudaHostAllocDefault));
        gpuErrorCheck(cudaHostAlloc((void **) &h_chunk_2, chunk_size, cudaHostAllocDefault));
        total_host_memory += 2 * chunk_size;
    } else {
        gpuErrorCheck(cudaHostAlloc((void **) &h_chunk_1, chunk_size, cudaHostAllocDefault));
        h_chunk_2 = h_chunk_1;
        total_host_memory += 1 * chunk_size;
    }


    if (benchmark_mode) {
    	verbose("Benchmark mode - filling host buffer with random data.\n");
    	for (int i = 0; i < info.bpp * info.w * info.h; i++) {
    		h_chunk_1[i] = static_cast<unsigned char>(rand() % 255);
    		h_chunk_2[i] = static_cast<unsigned char>(rand() % 255);
    	}
    }
    // work space (multi-stream)
    size_t workspace_size = sizeof(float) * chunk_frame_count * main_scale * main_scale;

    float *d_workspace_1;
    float *d_workspace_2;

    if (multistream) {
        gpuErrorCheck(cudaMalloc((void** ) &d_workspace_1, workspace_size));
        gpuErrorCheck(cudaMalloc((void** ) &d_workspace_2, workspace_size));
        total_device_memory += 2 * workspace_size;
    } else {
        gpuErrorCheck(cudaMalloc((void** ) &d_workspace_1, workspace_size));
        d_workspace_2= d_workspace_1;
        total_device_memory += 1 * workspace_size;
    }

    // FFT buffer
    size_t fft_buffer_size = 0;
    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];
        int tiles_per_frame = (main_scale / scale) * (main_scale / scale);
        int tile_size = (scale / 2 + 1) * scale;

        fft_buffer_size += sizeof(cufftComplex) * tile_size * tiles_per_frame * buffer_frame_count;
    }

    cufftComplex *d_fft_buffer;

    gpuErrorCheck(cudaMalloc((void** ) &d_fft_buffer, fft_buffer_size));

    total_device_memory += fft_buffer_size;

    // FFT intensity accumulator (multi-stream) - initial values set to zero
    size_t accum_size = 0;
    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];
        int tiles_per_frame = (main_scale / scale) * (main_scale / scale);
        accum_size += sizeof(float) * (scale / 2 + 1) * scale * tiles_per_frame * tau_count;
    }

    float *d_accum_1;
    float *d_accum_2;

    if (multistream) {
        gpuErrorCheck(cudaMalloc((void** ) &d_accum_1, accum_size));
        gpuErrorCheck(cudaMalloc((void** ) &d_accum_2, accum_size));

        gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
        gpuErrorCheck(cudaMemset(d_accum_2, 0, accum_size));

        total_device_memory += 2 * workspace_size;
    } else {
        gpuErrorCheck(cudaMalloc((void** ) &d_accum_1, accum_size));

        gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
        d_accum_2 = d_accum_1;

        total_device_memory += 1 * workspace_size;
    }

    // If we wish to scale the input unsinged chars we can do so using a lookup table
    // however in normal operation we don't scale so faster to use a conversion
//
//    float h_uchar_float_lookup[256];
//
//    for (int i = 0; i < 256; i++) {
//    	h_uchar_float_lookup[i] = static_cast<float>(i);
//    }
//    cudaMemcpyToSymbol(dk_uchar_float_lookup, h_uchar_float_lookup, sizeof(float)*256);


    size_t free_memory = 0;
    size_t total_memory = 0;
    gpuErrorCheck(cudaMemGetInfo(&free_memory, &total_memory));

    verbose("Memory allocation finished. "
            "Total memory allocated\n"
            "Device:\n\tExplictly allocated:\t %f GB \n\tTotal allocated:\t %f GP\n\tFree memory remaining:\t %f GB\n"
            "Host:\t %f GB\n"
            "info.bpp:\t %d\n",
            total_device_memory          / (float) 1073741824,
            (total_memory - free_memory) / (float) 1073741824,
            free_memory                  / (float) 1073741824,
            total_host_memory            / (float) 1073741824),
            info.bpp;


    // tau-vector
    int *d_tau_vector;
    gpuErrorCheck(cudaMalloc((void** ) &d_tau_vector, tau_count * sizeof(int)));
    gpuErrorCheck(cudaMemcpy(d_tau_vector, tau_vector, tau_count * sizeof(int), cudaMemcpyHostToDevice));
    total_device_memory += sizeof(int) * tau_count;

    verbose("Memory Allocations Done.\n");
    //////////
    ///  FFT Plan
    //////////

    cufftHandle *FFT_plan_list = new cufftHandle[scale_count];

    int rank = 2;
    int istride = 1;
    int ostride = 1;

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];

        int tiles_per_frame = (main_scale / scale) * (main_scale / scale);
        int batch_count = chunk_frame_count * tiles_per_frame;
        int n[2] = {scale, scale};

        int idist = scale * scale;
        int odist = scale * (scale/2+1);

        int inembed[] = {scale, scale};
        int onembed[] = {scale, scale/2+1};

        size_t mem_usage;

        verbose("FFT Plan Info:\n");
        verbose("\tn: (%d, %d), inembed: (%d, %d), onembed: (%d, %d), idist, odist: (%d, %d), batch: %d\n", n[0], n[1], inembed[0], inembed[1], onembed[0], onembed[1], idist, odist, batch_count);

        int plan_code = cufftPlanMany(&FFT_plan_list[s], rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch_count);
        int esti_code = cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch_count, &mem_usage);

        conditionAssert(plan_code == CUFFT_SUCCESS, "main cuFFT plan failure", true);
        conditionAssert(esti_code == CUFFT_SUCCESS, "error estimating cuFFT plan memory usage", true);
    }

    verbose("FFT Plan Done.\n");
    //////////
    ///  Pointer Allocation
    //////////

    // FFT'd buffer & FFT intensity accumulator are scale dependent so we define a array to hold values for each scale

    float **d_accum_list_1 = new float*[scale_count];
    float **d_accum_list_2 = new float*[scale_count];
    cufftComplex **d_fft_buffer_list    = new cufftComplex*[scale_count];

    d_accum_list_1[0] = d_accum_1;
    d_accum_list_2[0] = d_accum_2;
    d_fft_buffer_list[0] = d_fft_buffer;

    for (int s = 0; s < scale_count - 1; s++) {
        int scale = scale_vector[s];

        int tile_size = (scale/2 + 1) * scale;
        int tiles_per_frame = (main_scale / scale) * (main_scale / scale);

        d_accum_list_1[s+1] = d_accum_list_1[s] + tiles_per_frame * tile_size * tau_count;
        d_accum_list_2[s+1] = d_accum_list_2[s] + tiles_per_frame * tile_size * tau_count;

        d_fft_buffer_list[s+1] = d_fft_buffer_list[s] + tiles_per_frame * tile_size * buffer_frame_count;
    }

    cufftComplex **d_start_list = new cufftComplex*[scale_count];
    cufftComplex **d_end_list   = new cufftComplex*[scale_count];
    cufftComplex **d_junk_list  = new cufftComplex*[scale_count];

    for (int s = 0; s < scale_count; s++) {
        int tiles_per_frame = (main_scale / scale_vector[s]) * (main_scale / scale_vector[s]);
        int tile_size  = (scale_vector[s]/2 + 1) * scale_vector[s];

        d_start_list[s]  = d_fft_buffer_list[s];
        d_end_list[s]    = d_fft_buffer_list[s] + 1 * tiles_per_frame * tile_size * chunk_frame_count;
        d_junk_list[s]   = d_fft_buffer_list[s] + 2 * tiles_per_frame * tile_size * chunk_frame_count;
    }

    unsigned char *d_idle  = d_buffer;
    unsigned char *d_ready = d_buffer + 1 * chunk_frame_count * info.bpp * info.w * info.h;
    unsigned char *d_used  = d_buffer + 2 * chunk_frame_count * info.bpp * info.w * info.h;

    // pointers to shuffle with stream

    float *d_workspace_cur = d_workspace_1;
    float *d_workspace_nxt = d_workspace_2;

    float **d_accum_list_cur = d_accum_list_1;
    float **d_accum_list_nxt = d_accum_list_2;

    unsigned char *h_chunk_cur = h_chunk_1;
    unsigned char *h_chunk_nxt = h_chunk_2;

    cudaStream_t *stream_cur = &stream_1;
    cudaStream_t *stream_nxt = &stream_2;

    if (!multistream) {
        d_workspace_nxt = d_workspace_cur;
        d_accum_list_nxt = d_accum_list_cur;
        h_chunk_nxt = h_chunk_cur;
        stream_nxt = stream_cur;
    }

    verbose("Pointer Allocations Done\n");
    //////////
    ///  Main Loop
    //////////
    verbose("Main loop start.\n");

    // Initialise CPU memory (h_ready / idle)

    loadVideoToHost(use_moviefile, moviefile, cap, h_chunk_nxt, info, chunk_frame_count, benchmark_mode);
    loadVideoToHost(use_moviefile, moviefile, cap, h_chunk_cur, info, chunk_frame_count, benchmark_mode); // puts chunk data into pinned host memory
    verbose("Video loaded to host\n");
    gpuErrorCheck(cudaMemcpyAsync(d_idle, h_chunk_nxt, chunk_size, cudaMemcpyHostToDevice, *stream_cur));
    verbose("GPU error check\n");
    
    parseChunk(d_idle, d_start_list, d_workspace_cur, scale_vector, scale_count, chunk_frame_count, info, FFT_plan_list, *stream_cur);

    gpuErrorCheck(cudaStreamSynchronize(*stream_cur));
    verbose("Parse chunk\n");

    for (int chunk_index = 0; chunk_index < total_chunks; chunk_index++) {

        gpuErrorCheck(cudaMemcpyAsync(d_ready, h_chunk_cur, chunk_size, cudaMemcpyHostToDevice, *stream_cur));

        parseChunk(d_ready, d_end_list, d_workspace_cur, scale_vector, scale_count, chunk_frame_count, info, FFT_plan_list, *stream_cur);

        for (int frame_offset = 0; frame_offset < chunk_frame_count; frame_offset += 1) {
            analyseChunk(d_start_list, d_end_list, d_accum_list_cur, scale_count, scale_vector, leftover_frames, chunk_frame_count, frame_offset, tau_count, tau_vector, *stream_cur);
        }

        // prevent overrun
        gpuErrorCheck(cudaStreamSynchronize(*stream_nxt));

        if (total_chunks - chunk_index > 2) {
            loadVideoToHost(use_moviefile, moviefile, cap, h_chunk_nxt, info, chunk_frame_count, benchmark_mode);
        } else if (leftover_frames != 0 && total_chunks - chunk_index == 2) {
            loadVideoToHost(use_moviefile, moviefile, cap, h_chunk_nxt, info, leftover_frames, benchmark_mode);
        }

        //// Pointer swap

        swap<unsigned char>(h_chunk_cur, h_chunk_nxt);
        swap<float>(d_workspace_cur, d_workspace_nxt);
        swap<float*>(d_accum_list_cur, d_accum_list_nxt);
        swap<cudaStream_t>(stream_cur, stream_nxt);

        rotateThreePtr<cufftComplex*>(d_junk_list, d_start_list, d_end_list);
        rotateThreePtr<unsigned char>(d_used, d_ready, d_idle);

        // End of iteration
        verbose("[Chunk complete (%d \\ %d)]\n", chunk_index + 1, total_chunks);

        // dump_accum_after = 0 by default
        if (dump_accum_after != 0 && chunk_index != 0 && chunk_index % dump_accum_after == 0) {
            verbose("[Parsing Accumulator]\n");

            cudaDeviceSynchronize();
            if (multistream) {
                combineAccumulators(d_accum_list_cur, d_accum_list_nxt, scale_vector, scale_count, tau_count);
            }

            int tmp_frame_count = chunk_frame_count * dump_accum_after;

            std::string tmp_name = file_out + "_t" + std::to_string(chunks_already_parsed / dump_accum_after) + "_";

            // d_accum_list_cur should be the non-averaged FFT result 

            // analyse_accums() average the FFT result 
            analyse_accums(scale_vector, scale_count, lambda_arr, lambda_count, tau_vector, tau_count, tmp_frame_count, mask_tolerance, tmp_name, d_accum_list_cur, info.fps);

            verbose("[Purging Accumulator]\n");

            if (multistream) {
                gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));
                gpuErrorCheck(cudaMemset(d_accum_2, 0, accum_size));
            } else {
                gpuErrorCheck(cudaMemset(d_accum_1, 0, accum_size));

            }

            chunks_already_parsed += dump_accum_after;
        }
    }

    /// Extra frames
    if (leftover_frames != 0) {
        verbose("[+ %d extra frames]\n", leftover_frames);
        size_t extra_frames_size = sizeof(unsigned char) * leftover_frames * info.bpp * info.w * info.h;

        gpuErrorCheck(cudaMemcpyAsync(d_ready, h_chunk_cur, extra_frames_size, cudaMemcpyHostToDevice, *stream_cur));
        parseChunk(d_ready, d_end_list, d_workspace_cur, scale_vector, scale_count, leftover_frames, info, FFT_plan_list, *stream_cur);

        for (int frame_offset = 0; frame_offset < leftover_frames; frame_offset += 1) {
            analyseChunk(d_start_list, d_end_list, d_accum_list_cur, scale_count, scale_vector, leftover_frames, chunk_frame_count, frame_offset, tau_count, tau_vector, *stream_cur);
        }
    }
    cudaDeviceSynchronize();

    auto end_main = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < scale_count; s++) {
        cufftDestroy(FFT_plan_list[s]);
    }

    // Free memory locations we no longer need

    cudaFree(h_chunk_1);
    cudaFree(h_chunk_2);
    cudaFree(d_buffer);
    cudaFree(d_fft_buffer);
    cudaFree(d_workspace_1);
    cudaFree(d_workspace_2);

    if (multistream) {
        combineAccumulators(d_accum_list_cur, d_accum_list_nxt, scale_vector, scale_count, tau_count);
        //cudaFree(d_accum_2);
    }

    //////////
    ///  Analysis
    //////////analyseChunk
    verbose("Analysis.\n");

    int frames_left = total_frames - chunks_already_parsed * chunk_frame_count;
    analyse_accums(scale_vector, scale_count, lambda_arr, lambda_count, tau_vector, tau_count, frames_left, mask_tolerance, file_out, d_accum_list_cur, info.fps);

    cudaDeviceSynchronize();
    cudaFree(d_accum_1);

    auto end_out = std::chrono::high_resolution_clock::now();

    auto duration1 = std::chrono::duration_cast < std::chrono::microseconds
            > (end_main - start_time).count();
    auto duration2 = std::chrono::duration_cast < std::chrono::microseconds
            > (end_out - end_main).count();

    printf("[Time elapsed] "
           "\n\tMain:\t\t%f s, "
           "\n\tRadial Average:\t%f s,"
           "\n\tTotal\t\t%f s,\t(%f frame / second)\n",
           (float) duration1 / 1e6,
           (float) duration2 / 1e6,
           ((float) duration1 + (float) duration2) / 1e6,
           (float) (total_frames * 1e6) / ((float) duration1 + (float) duration2));
}
