#include <cuda_runtime.h>

#ifndef CONSTANTS_HPP_
#define CONSTANTS_HPP_

int const BLOCKSIZE_X = 16;  // Sometimes desirable to launch 2D block of threads
int const BLOCKSIZE_Y = 16;  // rather than 1D block
int const BLOCKSIZE = 256;

__constant__ float dk_uchar_float_lookup[256];

#endif /* CONSTANTS_HPP_ */
