// Copyright 2021 George Haskell (gh455)

#include <unistd.h>
#include <stdio.h>

#include <string>
#include <iostream>
#include <fstream>

#include "debug.hpp"

struct DDMparams {
	std::string     file_in;
	std::string     file_out;
	std::string     q_file_name;	// file-path for q-vector
	std::string     t_file_name;	// file-path for tau-vector
	std::string		s_file_name;	// file-path for scale-vector

	int    frame_count;				// number of frames to analyse
	int    x_off 			= 0;    // number of pixels to offset x=0 by in frame
	int    y_off 			= 0;	// number of pixels to offset y=0 by in frame
	int    chunk_length		= 30;   // number of frames in frame buffer
	int    rolling_purge	= 0;    // purge and analyse accumulators after number of frames

	bool   use_frame_rate	= true;
	bool   use_webcam 		= false;
	bool   use_movie_file	= false;
	int	   frame_rate;
	bool   explicit_fps		= false;
	int    webcam_idx 		= 0;
	bool   multi_stream 	= true;
	float  q_tolerence		= 1.2;
} params;

// forward declare main DDM function
void runDDM(std::string file_in,
            std::string file_out,
			int *tau_arr, 	int tau_count,
			float *q_arr,	int q_count,
			int *scale_arr, int scale_count,
			int x_off,		int y_off,
			int total_frames,
			int chunk_frame_count,
			bool multistream,
			bool use_webcam,
			int webcam_idx,
			float q_tolerance,
			bool is_movie_file,
			int frame_rate,
			int use_frame_rate,
			int dump_accum_after,
			bool use_explicit_frame_rate);

void printHelp() {
    fprintf(stderr,
    		"\n ~~ multiscale DDM - CUDA - HELP ~~ \n"
    		" - G. Haskell (gh455) 2021"
    		"\n"
			"  Usage ./ddm [OPTION]..\n"
			"  -h           Print out this help.\n"
			"   REQUIRED ARGS\n"
			"  -o PATH      Output filepath.\n"
			"  -N INT       Number of frames to analyse.\n"
			"  -Q PATH      Specify path to q-value file (line separated).\n"
			"  -T PATH 		Specify path to tau-value file (line separated). \n"
			"  -S PATH 		Specify path to scale-value file (line separated). \n"

    		"   INPUT ARGS\n"
			"  -f PATH      Specify path to input video (either -f or -W option must be given).\n"
			"  -W INT       Use web-camera as input video, (web-camera number can be supplied, defaults to first connected camera).\n"

			"   OPTIONAL ARGS\n"
			"  -x OFFSET        Set x-offset (default 0).\n"
			"  -y OFFSET        Set y-offset (default 0).\n"
			"  -I           	Use frame indices for tau-labels not real time.\n"
			"  -v			Verbose mode on.\n"
			"  -Z           Turn off multi-steam (smaller memory footprint - slower execution time).\n"
			"  -t INT       Set the q-vector mask tolerance - percent (integer only) (default 20 i.e. radial mask (1 - 1.2) * q).\n"
			"  -C INT	    Set main chunk frame count, a buffer 3x chunk frame count will be allocated in memory (default 30 frames).\n"
			"  -G SIZE          Sub-divide analysis, buffer will be output and purged every SIZE chunks\n"
    		"  -M FPS\n"
    		"  -F FPS ");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

	printf("DDM Start\n");

    // Flags
    bool input_specified = false;
    bool movie_file = false;

    for (;;) {
        switch (getopt(argc, argv, "ho:N:x:y:Q:T:S:If:W::vZt:C:M:G:F:")) {
            case '?':
            case 'h':
                printHelp();
                return -1;

            case 'o':
                params.file_out = optarg;
                continue;

            case 'N':
                params.frame_count = atoi(optarg);
                continue;

            case 'x':
                 params.x_off = atoi(optarg);
                 continue;

             case 'y':
                 params.y_off = atoi(optarg);
                 continue;

             case 'Q':
                 params.q_file_name = optarg;
                 continue;

             case 'T':
                 params.t_file_name = optarg;
                 continue;

             case 'S':
                 params.s_file_name = optarg;
                 continue;

             case 'I':
                 params.use_frame_rate = false;
                 continue;

            case 'f':
            	{
					conditionAssert(!input_specified, "Cannot use both in-filepath and web-cam option at same time.", true);

					params.file_in = optarg;
					input_specified = true;
            	}
                continue;

            case 'W':
            	{
					conditionAssert(!input_specified, "Cannot use both in-filepath and web-cam option at same time.", true);

					params.use_webcam = true;
					if (optarg != NULL) {
						params.webcam_idx = atoi(optarg);
					}
					input_specified = true;
            	}
                continue;

            case 'v':
                setVerbose(true);
                continue;

            case 'Z':
                params.multi_stream = false;
                continue;

            case 't':
            	{
            		int tmp = atoi(optarg);
            		params.q_tolerence = 1.0 + (tmp / 100.0);
            	}
            	continue;

            case 'C':
                params.chunk_length = atoi(optarg);
                continue;

            case 'M':
            	{
					params.use_movie_file = true;
					params.frame_rate = atoi(optarg);
            	}
                continue;

            case 'G':
                params.rolling_purge = atoi(optarg);
                continue;

            case 'F':
            	{
                params.explicit_fps = true;
                params.frame_rate = atoi(optarg);
            	}
                continue;

        }
        break;
    }

    if (optind != argc) {
        printHelp();
        conditionAssert(false, "An unexpected option was found.", true);
    }

    conditionAssert(input_specified, "Must specify input.", true);

    std::ifstream q_file(params.q_file_name);
    std::ifstream t_file(params.t_file_name);
    std::ifstream s_file(params.s_file_name);

    conditionAssert(q_file.is_open(), "cannot open q-file.", true);
    conditionAssert(t_file.is_open(), "cannot open tau-file", true);
    conditionAssert(s_file.is_open(), "cannot open scales-file.", true);

    // First count number of elements in each file

    float tmp_q;
    int q_count = 0;
    while (q_file >> tmp_q) {
        q_count++;
    }

    int tmp_tau;
    int tau_count = 0;
    while (t_file >> tmp_tau) {
        tau_count++;
    }

    int tmp_scale;
    int scale_count = 0;
    while (s_file >> tmp_scale) {
    	scale_count++;
    }

    // Seek back to the beginning of the file and read the values

    q_file.clear();
    q_file.seekg(0, std::ios::beg);
    t_file.clear();
    t_file.seekg(0, std::ios::beg);
    s_file.clear();
    s_file.seekg(0, std::ios::beg);

    float q_arr[q_count];
    int tau_arr[tau_count];
    int scale_arr[scale_count];

    int idx;

    idx = 0;
    while (q_file >> tmp_q) {
        q_arr[idx] = tmp_q;
        idx++;
    }

    idx = 0;
    while (t_file >> tmp_tau) {
        tau_arr[idx] = tmp_tau;
        idx++;
    }

    idx = 0;
    while (s_file >> tmp_scale) {
    	scale_arr[idx] = tmp_scale;
        idx++;
    }

    runDDM(params.file_in,
    	   params.file_out,
		   tau_arr,
		   tau_count,
		   q_arr,
		   q_count,
		   scale_arr,
		   scale_count,
		   params.x_off,
		   params.y_off,
		   params.frame_count,
		   params.chunk_length,
		   params.multi_stream,
		   params.use_webcam,
		   params.webcam_idx,
		   params.q_tolerence,
		   params.use_movie_file,
		   params.frame_rate,
		   params.frame_rate,
		   params.rolling_purge,
		   params.explicit_fps);

    printf("DDM End\n");
}



