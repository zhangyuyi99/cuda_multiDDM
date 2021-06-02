# cuda_multiDDM
A CUDA implementation of multi-scale DDM

Requires CUDA, OpenCV

 ~~ multiscale DDM - CUDA - HELP ~~ 

  Usage ./ddm [OPTION]..
  -h           Print out this help.
   REQUIRED ARGS
  -o PATH     Output filepath.
  -N INT      Number of frames to analyse.
  -Q PATH     Specify path to lambda-value file (line separated).
  -T PATH 		Specify path to tau-value file (line separated). 
  -S PATH 		Specify path to scale-value file (line separated). 

   INPUT ARGS
  -f PATH     Specify path to input video (either -f or -W option must be given).
  -W INT      Use web-camera as input video, (web-camera number can be supplied, defaults to first connected camera).
  -B 		      Benchmark mode, will perform analysis on random data.

   OPTIONAL ARGS
  -x OFFSET   Set x-offset (default 0).
  -y OFFSET   Set y-offset (default 0).
  -I          Use frame indices for tau-labels not real time.
  -v			    Verbose mode on.
  -Z          Turn off multi-steam (smaller memory footprint - slower execution time).
  -t INT      Set the q-vector mask tolerance - percent (integer only) (default 20 i.e. radial mask (1 - 1.2) * q).
  -C INT	    Set main chunk frame count, a buffer 3x chunk frame count will be allocated in memory (default 30 frames).
  -G SIZE     Sub-divide analysis, buffer will be output and purged every SIZE chunks.
  -M FPS		  Must be used if using movie-file file format. Argument to set frame-rate of movie-file.
  -F FPS 		  Force the analysis to assume a specific frame-rate, over-rides other options.
