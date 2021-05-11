// Copyright 2021 George Haskell (gh455)

#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>
#include <stdbool.h>
#include <unistd.h>

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


// If we choose to dual-stream the code then we must combine the FFT intensity accumulator
// associated with each stream.
inline void combineAccumulators(float **d_accum_list_A,
                                float **d_accum_list_B,
                                int *scale_vector,
                                int scale_count,
                                int tau_count) {


    dim3 blockDim(BLOCKSIZE);
    int main_scale = scale_vector[0];

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];
        int tile_count = (main_scale / scale) * (main_scale / scale);
        int frame_size = (scale / 2 + 1) * scale * tile_count;

        int gridDim = ceil(frame_size / static_cast<float>(BLOCKSIZE));

        combineAccum<<<gridDim, blockDim>>>(d_accum_list_A[s], d_accum_list_B[s], tau_count, frame_size);
    }
}


void analyse_accums(int *scale_vector,
                    int scale_count,
                    float *q_vector,
                    int q_count,
                    int *tau_vector,
                    int tau_count,
                    int frames_analysed,
                    float q_tolerance,
                    std::string file_out,
                    float **accum_list,
                    int frame_rate) {

    int main_scale = scale_vector[0];

    bool *d_masks;

    gpuErrorCheck(cudaMalloc((void** ) &d_masks, sizeof(bool) * (main_scale / 2 + 1) * main_scale * q_count))

    int *h_px_count = new int[q_count * scale_count]();
    float norm_factor = static_cast<float>(frames_analysed);

    float *q_vector_tmp = new float[q_count];
    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];
        int tile_count = (main_scale / scale) * (main_scale / scale);
        int tile_size = (scale / 2 + 1) * scale;

        for (int i = 0; i < q_count; i++) {
            q_vector_tmp[i] = q_vector[i] * (scale / static_cast<float>(main_scale));
        }

        buildAzimuthMask(d_masks, h_px_count, q_vector_tmp, q_count, q_tolerance, scale, scale);

        for (int tile_idx = 0; tile_idx < tile_count; tile_idx++) {
            int tile_offset = tile_size * tile_idx;

            std::string scale_name = file_out + std::to_string(scale) + "-" + std::to_string(tile_idx);

            float *d_accum_tmp = accum_list[s] + tile_offset;

            analyseFFTDevice(scale_name, d_accum_tmp, d_masks, h_px_count, norm_factor, tau_vector, tau_count, q_vector, q_count, tile_count, scale, scale, frame_rate);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! This function handles the parsing of on-device raw (uchar) data into a float
//! array, and the multi-scale FFT of this data to a list of cufftComplex arrays.
//! @param g_raw_in input video frames in global memory
//! @param g_fft_list_out, list of FFT frames for different length-scales
//! @param w frame width in pixels
//! @param h frame height in pixels
//! @param frame_count number of frames in chunk
//! @param scale_arr TODO
//! @param scale_count TODO
//! @param fft_plan_list TODO
//! @param stream Stream to run GPU kernels on
////////////////////////////////////////////////////////////////////////////////
void parseChunk(unsigned char *d_raw_in,
                cufftComplex **d_fft_list_out,
                float *d_workspace,
                int scale_count,
                int *scale_vector,
                int frame_count,
                video_info_struct info,
                cufftHandle *fft_plan_list,
                cudaStream_t stream) {

    int main_scale = scale_vector[0];

    int x_dim = static_cast<int>(ceil(main_scale / static_cast<float>(BLOCKSIZE_X)));
    int y_dim = static_cast<int>(ceil(main_scale / static_cast<float>(BLOCKSIZE_Y)));

    dim3 gridDim(x_dim, y_dim);
    dim3 blockDim(BLOCKSIZE_X, BLOCKSIZE_Y);

    for (int s = 0; s < scale_count; s++) {
        int scale = scale_vector[s];

        parseBufferScalePow2<<<gridDim, blockDim, 0, stream>>>(d_raw_in, d_workspace, info.bpp, 0, info.w, info.h, info.x_off, info.y_off, scale, main_scale, frame_count);
        cufftSetStream(fft_plan_list[s], stream);

        int exe_code = cufftExecR2C(fft_plan_list[s], d_workspace, d_fft_list_out[s]);
        conditionAssert(exe_code == CUFFT_SUCCESS, "cuFFT execution failure", true);
    }
}


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
        float fft_norm = 1.0f / px_count;

        dim3 gridDim(static_cast<int>(ceil(frame_size / static_cast<float>(BLOCKSIZE))));

        int frames_left = chunk_frame_count - frame_offset;

        cufftComplex *tmp;

        for (int t = 0; t < tau_count; t++) {
            if (tau_vector[t] < frames_left) {
                tmp = d_fft_buffer1[s] + (frame_offset + tau_vector[t]) * frame_size;
            } else {
                tmp = d_fft_buffer2[s] + (tau_vector[t] - frames_left) * frame_size;
            }

            float *accum_out = d_fft_accum_list[s] + frame_size * t;

            processFFT<<<gridDim, blockDim, 0, stream>>>(d_fft_buffer1[s] + frame_size * frame_offset, tmp, accum_out, fft_norm, frame_size);
        }
    }
}


void runDDM(std::string file_in,
            std::string file_out,
            int *tau_vector,	int tau_count,
            float *q_vector, 	int q_count,
            int *scale_vector,  int scale_count,
            int x_offset, 		int y_offset,
            int total_frames,
            int chunk_frame_count,
            bool multistream,
            bool use_webcam,
            int webcam_idx,
            float q_tolerance,
            bool is_movie_file,
            int movie_frame_rate,
            int use_frame_rate,
            int dump_accum_after) {

    auto start_time = std::chrono::high_resolution_clock::now();
    verbose("[multiDDM Begin]\n");

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
                    cv::Rect rect(y_offset + (t / tiles_per_side) * scale + s, x_offset + (t % tiles_per_side) * scale + s, scale, scale); // add s to each to help with readibility
                    cv::rectangle(tmp_frame, rect, cv::Scalar(0, 255, 0));
                }
            }
            cv::imshow( "Webcam", tmp_frame );

            char c=(char) cv::waitKey(25);
            if(c==27)
                break;
        }
        tmp_cap.release();
        cv::destroyAllWindows();
    }


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
    std::sort(q_vector, q_vector + q_count);
    std::sort(scale_vector, scale_vector + scale_count, std::greater<int>());

    //////////
    ///  Parameter Check
    //////////

    verbose("Scale list:\n");
    for (int s = 0; s < scale_count; s++) {
        unsigned int scale = scale_vector[s];
        printf("\t%d\n", scale);
        conditionAssert(!(scale == 0) && !(scale & (scale - 1)), "scales must be powers of two (> 0)", true);

        if (s < scale_count - 1)
            conditionAssert((scale_vector[s] > scale_vector[s + 1]), "scales should be descending order", true);
    }

    for (int t = 0; t < tau_count - 1; t++) {
        conditionAssert((tau_vector[t] >= 0), "Tau values should be positive", true);
        conditionAssert((tau_vector[t] < tau_vector[t + 1]), "Tau vector should be ascending order", true);
    }

    for (int q = 0; q < q_count - 1; q++) {
        conditionAssert((q_vector[q] >= 0), "q-vector values should be positive", true);
        conditionAssert((q_vector[q] < q_vector[q + 1]), "q-vector vector should be ascending order", true);
    }

    conditionAssert(q_tolerance < 10 && q_tolerance > 1.0,
            "q_tolerance is likely undesired value, refer to README for more information");

    conditionAssert(q_vector[q_count-1] <= scale_vector[scale_count-1],
            "The largest q-vector should be smaller than the smallest scale.", true);

    verbose("%d tau-values.\n", tau_count);
    verbose("%d q-vector values.\n", q_count);

    verbose("Parameter Check Done.\n");
    //////////
    ///  Video Setup
    //////////

    video_info_struct info;
    FILE *moviefile;
    cv::VideoCapture cap;

    int frame_rate;

    if (is_movie_file) { // if we have a .moviefile folder we open with own custom reader
        moviefile = fopen(file_in.c_str(), "rb");
        conditionAssert(moviefile != NULL, "couldn't open .movie file", true);

        info = initFile(moviefile);
        frame_rate = movie_frame_rate;

    } else {
        if (use_webcam) {
            cap = cv::VideoCapture(webcam_idx);
        } else {
            cap = cv::VideoCapture(file_in);
        }

        conditionAssert(cap.isOpened(), "error opening video file with openCV", true);

        info.w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        info.h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        frame_rate = cap.get(cv::CAP_PROP_FPS);

        cv::Mat test_img;
        cap >> test_img;

        // Due to the difficulty in dealing with many image types, we only
        // deal with multi-channel data if image is CV_8U (i.e. uchar)
        int type = test_img.type();
        info.bpp = (type % 8) ? 1 : test_img.channels();

        if (!use_webcam)
            cap = cv::VideoCapture(file_in); // re-open so can view first frame again
    }

    if (!use_frame_rate) {
        frame_rate = 1;  // using raw tau indices is same as FPS = 1
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

    verbose("[Video info - (%d x %d), %d Frames, %d FPS]\n", info.w, info.h, total_frames, frame_rate);

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

    // main device buffer
    size_t buffer_size  = sizeof(unsigned char) * buffer_frame_count * info.bpp * info.w * info.h;

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

    // uchar to float conversion table

    float h_uchar_float_lookup[256];

    for (int i = 0; i < 256; i++) {
    	h_uchar_float_lookup[i] = static_cast<float>(i);
    }
    cudaMemcpyToSymbol(dk_uchar_float_lookup, h_uchar_float_lookup, sizeof(float)*256);


    size_t free_memory = 0;
    size_t total_memory = 0;
    gpuErrorCheck(cudaMemGetInfo(&free_memory, &total_memory));

    verbose("Memory allocation finished. "
            "Total memory allocated\n"
            "Device:\n\tExplictly allocated:\t %f GB \n\tTotal allocated:\t %f GP\n\tFree memory remaining:\t %f GB\n"
            "Host:\t %f GB\n",
            total_device_memory          / (float) 1073741824,
            (total_memory - free_memory) / (float) 1073741824,
            free_memory                  / (float) 1073741824,
            total_host_memory            / (float) 1073741824);


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

    // FFT'd buffer & FFT intensity accumulator are scale dependent

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

    loadVideoToHost(is_movie_file, moviefile, cap, h_chunk_nxt, info, chunk_frame_count);
    loadVideoToHost(is_movie_file, moviefile, cap, h_chunk_cur, info, chunk_frame_count); // puts chunk data into pinned host memory

    gpuErrorCheck(cudaMemcpyAsync(d_idle, h_chunk_nxt, chunk_size, cudaMemcpyHostToDevice, *stream_cur));

    parseChunk(d_idle, d_start_list, d_workspace_cur, scale_count, scale_vector, chunk_frame_count, info, FFT_plan_list, *stream_cur);

    gpuErrorCheck(cudaStreamSynchronize(*stream_cur));


    for (int chunk_index = 0; chunk_index < total_chunks; chunk_index++) {

        gpuErrorCheck(cudaMemcpyAsync(d_ready, h_chunk_cur, chunk_size, cudaMemcpyHostToDevice, *stream_cur));

        parseChunk(d_ready, d_end_list, d_workspace_cur, scale_count, scale_vector, chunk_frame_count, info, FFT_plan_list, *stream_cur);

        for (int frame_offset = 0; frame_offset < chunk_frame_count; frame_offset += 1) {
            analyseChunk(d_start_list, d_end_list, d_accum_list_cur, scale_count, scale_vector, leftover_frames, chunk_frame_count, frame_offset, tau_count, tau_vector, *stream_cur);
        }

        // prevent overrun
        gpuErrorCheck(cudaStreamSynchronize(*stream_nxt));

        if (total_chunks - chunk_index > 2) {
            loadVideoToHost(is_movie_file, moviefile, cap, h_chunk_nxt, info, chunk_frame_count);
        } else if (leftover_frames != 0 && total_chunks - chunk_index == 2) {
            loadVideoToHost(is_movie_file, moviefile, cap, h_chunk_nxt, info, leftover_frames);
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

        if (dump_accum_after != 0 && chunk_index != 0 && chunk_index % dump_accum_after == 0) {
            verbose("[Parsing Accumulator]\n");

            cudaDeviceSynchronize();
            if (multistream) {
                combineAccumulators(d_accum_list_cur, d_accum_list_nxt, scale_vector, scale_count, tau_count);
            }

            int tmp_frame_count = chunk_frame_count * dump_accum_after;

            std::string tmp_name = file_out + "_t" + std::to_string(chunks_already_parsed / dump_accum_after) + "_";

            analyse_accums(scale_vector, scale_count, q_vector, q_count, tau_vector, tau_count, tmp_frame_count, q_tolerance, tmp_name, d_accum_list_cur, frame_rate);

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
        parseChunk(d_ready, d_end_list, d_workspace_cur, scale_count, scale_vector, leftover_frames, info, FFT_plan_list, *stream_cur);

        for (int frame_offset = 0; frame_offset < leftover_frames; frame_offset += 1) {
            analyseChunk(d_start_list, d_end_list, d_accum_list_cur, scale_count, scale_vector, leftover_frames, chunk_frame_count, frame_offset, tau_count, tau_vector, *stream_cur);
        }
    }
    cudaDeviceSynchronize();

    auto end_main = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < scale_count; s++) {
        cufftDestroy(FFT_plan_list[s]);
    }

    cudaFree(h_chunk_1);
    cudaFree(h_chunk_2);
    cudaFree(d_buffer);
    cudaFree(d_fft_buffer);
    cudaFree(d_workspace_1);
    cudaFree(d_workspace_2);

//    delete FFT_plan_list;
//    delete d_accum_list_1;
//    delete d_accum_list_2;
//    delete d_start_list;
//    delete d_junk_list;
//    delete d_end_list;

    if (multistream) {
        combineAccumulators(d_accum_list_cur, d_accum_list_nxt, scale_vector, scale_count, tau_count);
        //cudaFree(d_accum_2);
    }

    //////////
    ///  Analysis
    //////////
    verbose("Analysis.\n");

    int frames_left = total_frames - chunks_already_parsed * chunk_frame_count;
    analyse_accums(scale_vector, scale_count, q_vector, q_count, tau_vector, tau_count, frames_left, q_tolerance, file_out, d_accum_list_cur, frame_rate);

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
