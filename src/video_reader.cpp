// Copyright 2021 Jurij Kotar (jk400) & George Haskell (gh455)
/*
 Video Reader

 This file provides functionality to read a variety of video formats into memory.
 Using openCV a wide variety of video sources can be managed (including handles
 to direct video streams such as web-cameras. Additionally functionality to deal
 with .moviefiles is included. .moviefile functionality is heavily based on the
 work of Jurij Kotar.

 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <nvToolsExt.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <fstream>

#include "video_reader.hpp"
#include "debug.hpp"

void loadMovieToHost(FILE *moviefile,        unsigned char *h_buffer, video_info_struct info, int frame_count);
void loadCaptureToHost(cv::VideoCapture cap, unsigned char *h_buffer, video_info_struct info, int frame_count);

////////////////////////////////////////////////////////////////////////////////
//  GENERAL LOADER
////////////////////////////////////////////////////////////////////////////////

void loadVideoToHost(bool is_movie_file, FILE *moviefile, cv::VideoCapture capture, unsigned char *h_buffer, video_info_struct info, int frame_count) {
    if (is_movie_file) {
        loadMovieToHost(moviefile, h_buffer, info, frame_count);
    } else {
        loadCaptureToHost(capture, h_buffer, info, frame_count);
    }
}

////////////////////////////////////////////////////////////////////////////////
// .MOVIEFILE
////////////////////////////////////////////////////////////////////////////////

// Common camera defines

#define CAMERA_MOVIE_MAGIC 0x496d6554 // TemI
#define CAMERA_MOVIE_VERSION 1
#define CAMERA_TYPE_IIDC	1
#define CAMERA_TYPE_ANDOR	2
#define CAMERA_TYPE_XIMEA	3
#define CAMERA_PIXELMODE_MONO_8		1
#define CAMERA_PIXELMODE_MONO_16BE	2 // Big Endian
#define CAMERA_PIXELMODE_MONO_16LE	3 // Little Endian

#define IIDC_MOVIE_HEADER_LENGTH 172
#define ANDOR_MOVIE_HEADER_LENGTH 128
#define XIMEA_MOVIE_HEADER_LENGTH 240

video_info_struct initFile(FILE *moviefile) {
    /* Before accessing the .moviefile data we must first parse the header
     * Generic header information is first extracted, followed by specific
     * camera data.
     */

    // Locate start of header using magic value
    long magic_offset = 0;
    bool found = false;
    uint32_t magic_val;

    while (fread(&magic_val, sizeof(uint32_t), 1, moviefile) == 1) {
        if (magic_val == CAMERA_MOVIE_MAGIC) {
            found = true;
            break;
        }
        magic_offset++;
        fseek(moviefile, magic_offset, SEEK_SET);
    }

    fseek(moviefile, magic_offset, SEEK_SET);

    // Extract generic frame information
    camera_save_struct camera_frame;

    if (fread(&camera_frame, sizeof(struct camera_save_struct), 1, moviefile)
            != 1) {
        fprintf(stderr,
                "[.moviefile Initialise Error] Corrupted header at offset %lu\n",
                ftell(moviefile));
        exit(EXIT_FAILURE);
    }

    // Check read is as expected
    if (camera_frame.magic != CAMERA_MOVIE_MAGIC) {
        fprintf(stderr,
                "[.moviefile Initialise Error] Wrong magic at offset %lu\n",
                ftell(moviefile));
        exit(EXIT_FAILURE);
    }

    if (camera_frame.version != CAMERA_MOVIE_VERSION) {
        fprintf(stderr,
                "[.moviefile Initialise Error] Unsupported movie version %u\n",
                camera_frame.version);
        exit(EXIT_FAILURE);
    }

    // Return to file start to read specific camera information
    fseek(moviefile, -sizeof(struct camera_save_struct), SEEK_CUR);

    // Extract camera specific frame information
    uint32_t size_x, size_y;

    switch (camera_frame.type) {
    case CAMERA_TYPE_IIDC:
        struct iidc_save_struct iidc_frame;

        if (fread(&iidc_frame, IIDC_MOVIE_HEADER_LENGTH, 1, moviefile) != 1) {
            fprintf(stderr,
                    "[.moviefile Initialise Error] Corrupted header at offset %lu\n",
                    ftell(moviefile));
            exit(EXIT_FAILURE);
        }
        size_x = iidc_frame.i_size_x;
        size_y = iidc_frame.i_size_y;

        fseek(moviefile, -IIDC_MOVIE_HEADER_LENGTH, SEEK_CUR);
        break;

    case CAMERA_TYPE_ANDOR:
        struct andor_save_struct andor_frame;

        if (fread(&andor_frame, ANDOR_MOVIE_HEADER_LENGTH, 1, moviefile) != 1) {
            fprintf(stderr,
                    "[.moviefile Initialise Error] Corrupted header at offset %lu\n",
                    ftell(moviefile));
            exit(EXIT_FAILURE);
        }
        size_x = (andor_frame.a_x_end - andor_frame.a_x_start + 1)
                / andor_frame.a_x_bin;
        size_y = (andor_frame.a_y_end - andor_frame.a_y_start + 1)
                / andor_frame.a_y_bin;

        fseek(moviefile, -ANDOR_MOVIE_HEADER_LENGTH, SEEK_CUR);
        break;

    case CAMERA_TYPE_XIMEA:
        struct ximea_save_struct ximea_frame;

        if (fread(&ximea_frame, XIMEA_MOVIE_HEADER_LENGTH, 1, moviefile) != 1) {
            fprintf(stderr,
                    "[.moviefile Initialise Error] Corrupted header at offset %lu\n",
                    ftell(moviefile));
            exit(EXIT_FAILURE);
        }
        size_x = ximea_frame.x_size_x;
        size_y = ximea_frame.x_size_y;

        fseek(moviefile, -XIMEA_MOVIE_HEADER_LENGTH, SEEK_CUR);
        break;

    default:
        fprintf(stderr, "[.moviefile Initialise Error] Unsupported camera.\n");
        exit( EXIT_FAILURE);
        break;

    }

    video_info_struct out;
    out.w = static_cast<int>(size_x);
    out.h = static_cast<int>(size_y);
    out.type = camera_frame.type;
    out.length = camera_frame.length_data;
    out.bpp = (camera_frame.pixelmode == CAMERA_PIXELMODE_MONO_8) ? 1 : 2;

    return out;
}

/* We use camera information from the initialise phase, technically each frame could be taken
 * by a different camera, however this is highly unlikely, choose to throw an error if frame header
 * unexpected than repeat slow file manipulation from initialise phase.
 */
void loadMovieToHost(FILE *moviefile, unsigned char *h_buffer, video_info_struct info, int frame_count) {
    nvtxRangePush(__FUNCTION__); // Nvidia profiling option (for use in nvvp)

    int frame_index = 0;

    while (frame_index < frame_count) {
        // Read header
        switch (info.type) {
			case CAMERA_TYPE_IIDC:
				struct iidc_save_struct iidc_frame;
				if (fread(&iidc_frame, IIDC_MOVIE_HEADER_LENGTH, 1, moviefile) != 1) {
					fprintf(stderr, "[.moviefile Read Error] Corrupted header at offset %lu\n", ftell(moviefile));
					exit(EXIT_FAILURE);
				}

				break;

			case CAMERA_TYPE_ANDOR:
				struct andor_save_struct andor_frame;
				if (fread(&andor_frame, ANDOR_MOVIE_HEADER_LENGTH, 1, moviefile) != 1) {
					fprintf(stderr, "[.moviefile Read Error] Corrupted header at offset %lu\n", ftell(moviefile));
					exit(EXIT_FAILURE);
				}

				break;

			case CAMERA_TYPE_XIMEA:
				struct ximea_save_struct ximea_frame;
				if (fread(&ximea_frame, XIMEA_MOVIE_HEADER_LENGTH, 1, moviefile) != 1) {
					fprintf(stderr, "[.moviefile Read Error] Corrupted header at offset %lu\n", ftell(moviefile));
					exit(EXIT_FAILURE);
				}

				break;

			default:
				fprintf(stderr, "[.moviefile Read Error] Unsupported camera.\n");
				exit( EXIT_FAILURE);
				break;
			}

        unsigned char *h_current = h_buffer + info.w * info.h * info.bpp * frame_index;

        // Read data
        if (fread(h_current, info.length, 1, moviefile) != 1) {
            fprintf(stderr, "[.moviefile Read Error] Corrupted data at offset %lu\n", ftell(moviefile));
            exit(EXIT_FAILURE);
        }
        frame_index++;
    }
    nvtxRangePop();
}

////////////////////////////////////////////////////////////////////////////////
// openCV Video Reader
////////////////////////////////////////////////////////////////////////////////


//	This function takes a openCV video capture and loads a specified number of frames into a given uchar
//	pointer. The pointer must have a size at least [img.cols * img.rows * img.channels * frame_count]
//	If the frame is continuous in memory and each element is a uchar (i.e. Mat type 0, 8, 16, 24) then we
//	can do a direct memory copy. However if the image type is more complicated we do a much more time costly
//	iteration over the whole image. As we deal with uchars only - can lose image fidelity!

void loadCaptureToHost(cv::VideoCapture cap, unsigned char *h_buffer, video_info_struct info, int frame_count) {
    nvtxRangePush(__FUNCTION__);  // Nvidia profiling option (for use in nvvp)

    cv::Mat img;

    for (int frame_idx = 0; frame_idx < frame_count; frame_idx++) { // iterate over frame count
        cap >> img;  // move next video frame into image buffer

        conditionAssert(!img.empty(), " openVC video frame is empty.");

        int base_type = img.type() % 8;

        if (img.isContinuous() && base_type == 0) {
            // We have the simplest case that the image is a (multi-channel) uchar array in continuous memory
            int frame_bytes = img.cols * img.rows * img.elemSize();
            memcpy(h_buffer + frame_idx * frame_bytes, img.data, frame_bytes);
        } else {
            // If not simple case then we must directly iterate across image data array.
            // Slight speed up as only need consider out image dimensions (not full frame)
            // TODO: Not heavily tested as these video types not common.
            int width = info.w;
            int height = info.h;
            int frame_elements = img.cols * img.rows;

            int idx, oidx;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    idx = ((img.step) / img.elemSize1()) * y
                            + img.channels() * x;
                    oidx = frame_idx * frame_elements + y * img.cols + x;

                    switch (base_type) {// Note funky {} to allow redeclare of "data" var-name
                    case 1:  // CV_8S
                    {
                        char data = img.at<char>(y, x);
                        conditionAssert(data < 256, "pixel value too large to be casted to unsigned char.");
                        h_buffer[oidx] = static_cast<unsigned char>(data);
                    }
                        break;
                    case 2:  // CV_16U
                    {
                        unsigned short data = img.at<unsigned short>(y, x);
                        conditionAssert(data < 256, "pixel value too large to be casted to unsigned char.");
                        h_buffer[oidx] = static_cast<unsigned char>(data);
                    }
                        break;
                    case 3:  // CV_16S
                    {
                        short data = img.at<short>(y, x);
                        conditionAssert(data < 256, "pixel value too large to be casted to unsigned char.");
                        h_buffer[oidx] = static_cast<unsigned char>(data);
                    }
                        break;
                    case 4:  // CV_32S
                    {
                        int data = img.at<int>(y, x);
                        conditionAssert(data < 256, "pixel value too large to be casted to unsigned char.");
                        h_buffer[oidx] = static_cast<unsigned char>(data);
                    }
                        break;
                    case 5:  // CV_32F
                    {
                        float data = img.at<float>(y, x);
                        conditionAssert(data < 256, "pixel value too large to be casted to unsigned char.");
                        h_buffer[oidx] = static_cast<unsigned char>(data);

                    }
                        break;
                    case 6:  // CV_64F
                    {
                        double data = img.at<double>(y, x);
                        conditionAssert(data < 256, "pixel value too large to be casted to unsigned char.");
                        h_buffer[oidx] = static_cast<unsigned char>(data);
                    }
                        break;
                    }
                }
            }
        }
    }

    nvtxRangePop();
}

