// Copyright 2021 George Haskell (gh455)

#include <unistd.h>
#include <stdio.h>

#include <string>
#include <iostream>
#include <fstream>

#include "debug.hpp"

void runDDM(std::string file_in,
        std::string file_out,
        int tau_count,
        int *tau_vector,
        int q_count,
        float *q_vector,
        int scale_count,
        int *scale_vector,
        int x_offset,
        int y_offset,
        int total_frames,
        int chunk_frame_count,
        bool multistream,
        bool use_webcam,
        int webcam_idx,
        float q_tolerance,
        bool is_movie_file,
        int movie_frame_rate,
        int use_frame_rate,
        int dump_accum_after);

void printHelp() {
    fprintf(stderr,
            "\n     ## CUDA DDM HELP ## \n"
                    "  gh455 2021\n"
                    "\n"
                    "  Usage ./ddm [OPTION]..\n"
                    "  -h           Print out this help.\n"
                    "   REQUIRED ARGS\n"
                    "  -o PATH      Specify output path.\n"
                    "  -N INT       Specify number of frames.\n"
                    "  -q PATH      Specify path to q-value file (line separated).\n"
                    "  -t PATH          Specify path to tau-value file (line separated). \n"
                    "\n"
                    "   OPTIONAL ARGS\n"
                    "  -x SIZE      Set analysis width (defaults to image width).\n"
                    "  -y SIZE      Set analysis height (defaults to image height).\n"
                    "  -f PATH      Specify path to input video (either -f or -W option must be given).\n"
                    "  -W INT       Use web-camera as input video, (web-camera number can be suplied, defaults to first connected camera).\n"
                    "  -v           Set verbose mode.\n"
                    "  -C SIZE      Set main chunk frame count, a buffer 3x chunk frame count will be allocated in memory (default 50 frames).\n"
                    "  -Z           Turn off multi-steam (smaller memory footprint - slower execution time).\n"
                    "  -Q INT       Set the q-vector mask tolerance - percent (integer only) (default 20 i.e. radial mask (1 - 1.2) * q).\n"
                    "  -i OFFSET        Set x-offset (default 0).\n"
                    "  -j OFFSET        Set y-offset (default 0).\n"
                    "  -s STEP      Set under-sampling step size (default 1).\n"
                    "  -r REPEATS       Set under-sampling repeat count (default 1).\n"
                    "  -M FRAMERATE     Set input type to .moviefile (must specify integer frame rate).\n"
                    "  -F           Use frame counts for tau-spacing (not 1 / FPS).\n"
                    "  -G SIZE          Sub-divide analysis, buffer will be outputed and purged every SIZE chunks\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    struct DDMparams {
        std::string     file_in;
        std::string     file_out;
        std::string     q_file_name;
        std::string     t_file_name;

        int    frame_count;
        int    x_offset;
        int    y_offset;
        int    step_size;
        int    repeat_count;
        int    buffer_length;
        int    purge_cnum;

        int    frame_rate;
        bool            use_frame_rate;
        bool            use_webcam;
        int             webcam_idx;
        bool            multi_stream;
        float           q_tolerence;
    };


    printf("DDM Start\n");

    DDMparams params {
        .x_offset       = 0,
        .y_offset       = 0,
        .step_size      = 1,
        .repeat_count   = 1,
        .buffer_length  = 50, // 92 / 3
        .purge_cnum     = 0,
        .use_frame_rate = true,
        .use_webcam     = false,
        .webcam_idx     = -1,
        .multi_stream   = true,
        .q_tolerence    = 1.2,
    };

    // Flags
    bool supplied_input = false;
    bool movie_file = false;

    for (;;) {
        switch (getopt(argc, argv, "hf:W::o:N:q:t:vC:ZQ:i:j:s:r:M:FG:")) {
            case '?':
            case 'h':
                printHelp();
                return -1;

            case 'f':
                params.file_in = optarg;
                supplied_input = true;
                continue;

            case 'W':
                params.use_webcam = true;
                if (optarg == NULL) {
                    params.webcam_idx = 0;
                } else {
                    params.webcam_idx = atoi(optarg);
                }
                supplied_input = true;
                continue;

            case 'o':
                params.file_out = optarg;
                continue;

            case 'N':
                params.frame_count = atoi(optarg);
                continue;

            case 'q':
                params.q_file_name = optarg;
                continue;

            case 't':
                params.t_file_name = optarg;
                continue;
            case 'v':
                setVerbose(true);
                continue;

            case 'C':
                params.buffer_length = atoi(optarg) * 3;
                continue;

            case 'Z':
                params.multi_stream = false;
                continue;

            case 'Q':
                {
                    int tmp = atoi(optarg);
                    params.q_tolerence = 1.0 + (tmp / 100.0);
                }
                continue;

            case 'i':
                params.x_offset = atoi(optarg);
                continue;

            case 'j':
                params.y_offset = atoi(optarg);
                continue;

            case 's':
                params.step_size = atoi(optarg);
                continue;

            case 'r':
                params.repeat_count = atoi(optarg);
                continue;

            case 'M':
                {
                    movie_file = true;
                    params.frame_rate = atoi(optarg);
                }
                continue;
            case 'F':
                params.use_frame_rate = false;
                continue;

            case 'G':
                params.purge_cnum = atoi(optarg);
                continue;
        }
        break;
    }

    if (optind != argc) {
        printHelp();
        conditionAssert(false, "an unexpected option was found.", true);
    }

    conditionAssert(supplied_input, "must specify input.", true);

    int t;
    float q;
    std::ifstream q_file(params.q_file_name);
    std::ifstream t_file(params.t_file_name);

    conditionAssert(q_file.is_open(), "cannot open q-file.", true);
    conditionAssert(t_file.is_open(), "cannot open tau-file", true);


    int q_count = 0;
    while (q_file >> q) {
        q_count++;
    }

    int tau_count = 0;
    while (t_file >> t) {
        tau_count++;
    }

    q_file.clear();
    q_file.seekg(0, std::ios::beg);
    t_file.clear();
    t_file.seekg(0, std::ios::beg);

    float q_vector[q_count];
    int tau_vector[tau_count];

    int idx = 0;
    while (q_file >> q) {
        q_vector[idx] = q;
        idx++;
    }

    idx = 0;
    while (t_file >> t) {
        tau_vector[idx] = t;
        idx++;
    }

    int scale_count = 3;
    int scale_vector[scale_count] = {1024, 512, 256};


    runDDM(params.file_in,
           params.file_out,
           tau_count,
           tau_vector,
           q_count,
           q_vector,
           scale_count,
           scale_vector,
           params.x_offset,
           params.y_offset,
           params.frame_count,
           params.buffer_length/3,
           params.multi_stream,
           params.use_webcam,
           params.webcam_idx,
           params.q_tolerence,
           movie_file,
           params.frame_rate,
           params.use_frame_rate,
           0);

//    run(params.file_in,
//            params.file_out,
//            movie_file,
//            params.x_offset,
//            params.y_offset,
//            params.frame_count,
//            params.repeat_count,
//            params.step_size,
//            scale_count,
//            scale_vector,
//            q_count,
//            q_vector,
//            tau_count,
//            tau_vector,
//            params.q_tolerence,
//            params.buffer_length,
//            params.multi_stream,
//            params.frame_rate,
//            params.use_frame_rate,
//            params.use_webcam,
//            params.webcam_idx,
//            params.purge_cnum);

    printf("DDM End\n");
}



